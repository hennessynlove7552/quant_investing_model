"""
ML 기반 주가 예측 모델 구현.

입력: 각 종목별 과거 lookback일의 (일수익률, turnover proxy) 시퀀스를 평탄화하거나
      딥모델에서는 (L, lookback, 2) 텐서로 사용.
타깃: 다음날 수익률 (회귀) 또는 상위/하위 분위 이진 분류.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quantformer import _turnover_proxy, zscore_timesteps


def _panels(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """close (T,L), ret (T,L), v (T,L)"""
    prices = prices.sort_index().astype(float)
    close = prices.to_numpy()
    T, L = close.shape
    eps = 1e-12
    ret = np.zeros_like(close, dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps)
    vol_arr = None
    if volume is not None and not volume.empty:
        vol_arr = volume.reindex(prices.index).reindex(columns=prices.columns).astype(float).to_numpy()
    v = _turnover_proxy(ret.astype(np.float32), vol_arr)
    return close, ret.astype(np.float32), v


def _train_test_split_time(
    X: np.ndarray, y: np.ndarray, train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.3 < float(train_ratio) < 0.95):
        raise ValueError(f"train_ratio out of range: {train_ratio}")
    n = len(X)
    if n < 10:
        raise ValueError(f"not enough samples: {n}")
    split = int(n * float(train_ratio))
    split = max(5, min(n - 3, split))
    return X[:split], y[:split], X[split:], y[split:]


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(yt[m]) == np.sign(yp[m])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return float("nan")
    denom = np.maximum(np.abs(yt[m]), eps)
    return float(np.mean(np.abs((yt[m] - yp[m]) / denom)))


def build_univariate_return_dataset(
    prices_1: pd.Series,
    volume_1: Optional[pd.Series],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 티커용 데이터셋.
    X: (N, lookback, 2)  [ret, turnover] zscore
    y: (N,) next-day return
    close: (T,) 종가 (예측 가격 계산용)
    """
    prices_df = prices_1.to_frame("x")
    vol_df = volume_1.to_frame("x") if volume_1 is not None else None
    close, ret, v = _panels(prices_df, vol_df, lookback)
    close = close[:, 0].astype(np.float64)
    r = ret[:, 0].astype(np.float64)
    vv = v[:, 0].astype(np.float64)

    Xs, ys = [], []
    T = len(r)
    for t in range(lookback, T - 1):
        rw = r[t - lookback : t]
        vw = vv[t - lookback : t]
        x2 = np.stack([rw, vw], axis=-1).astype(np.float32)
        x2 = zscore_timesteps(x2.reshape(1, lookback, 2)).reshape(lookback, 2)
        nxt = r[t + 1]
        if not np.isfinite(nxt) or not np.all(np.isfinite(x2)):
            continue
        Xs.append(x2)
        ys.append(float(nxt))
    if len(Xs) < 20:
        raise ValueError(f"학습 샘플이 너무 적습니다 ({len(Xs)}개). 기간을 늘려주세요.")
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.float32), close


def train_predict_univariate(
    prices_1: pd.Series,
    volume_1: Optional[pd.Series],
    *,
    model_type: str = "auto",
    lookback: int = 20,
    hidden: int = 32,
    num_layers: int = 1,
    epochs: int = 15,
    lr: float = 1e-3,
    train_ratio: float = 0.7,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """
    단일 티커 다음 거래일 수익률 예측 + 테스트 지표/시각화용 시계열 반환.
    model_type: auto | ridge | svm | rf | lstm | gru
    """
    mt = (model_type or "auto").strip().lower().replace(" ", "_")
    X, y, close = build_univariate_return_dataset(prices_1, volume_1, lookback)
    Xtr, ytr, Xte, yte = _train_test_split_time(X, y, train_ratio=train_ratio)

    chosen = mt
    best = None

    def _fit_eval_one(m: str) -> dict:
        if m in ("ridge", "svm", "rf"):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR

            Xtr2 = Xtr.reshape(len(Xtr), -1)
            Xte2 = Xte.reshape(len(Xte), -1)
            if m == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
            elif m == "svm":
                model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.01))
            else:
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
                )
            model.fit(Xtr2, ytr)
            pred_te = model.predict(Xte2).astype(np.float64)
            pred_next = float(model.predict(X[-1:].reshape(1, -1))[0])
            return {
                "model": model,
                "pred_test": pred_te,
                "pred_next_return": pred_next,
                "device": "cpu",
            }

        if m in ("lstm", "gru"):
            torch.manual_seed(seed)
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = _SeqRegressor(m, in_dim=2, hidden=hidden, num_layers=num_layers).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
            ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
            Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
            model.train()
            for _ in range(int(epochs)):
                opt.zero_grad(set_to_none=True)
                pred = model(Xtr_t)
                loss = nn.functional.mse_loss(pred, ytr_t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_te = model(Xte_t).detach().cpu().numpy().astype(np.float64)
                pred_next = float(
                    model(torch.tensor(X[-1:], dtype=torch.float32, device=dev)).cpu().item()
                )
            return {
                "model": model,
                "pred_test": pred_te,
                "pred_next_return": pred_next,
                "device": str(dev),
            }

        raise ValueError(f"unknown model_type: {m}")

    if chosen == "auto":
        # 간단한 에이전트: 후보들을 검증(=test) MSE로 비교 후 최적 선택
        candidates = ["ridge", "svm", "rf", "lstm", "gru"]
        for c in candidates:
            try:
                out = _fit_eval_one(c)
                mse = float(np.mean((out["pred_test"] - yte.astype(np.float64)) ** 2))
                payload = {"candidate": c, "mse": mse, **out}
                if best is None or mse < best["mse"]:
                    best = payload
            except Exception:
                continue
        if best is None:
            raise ValueError("auto 모델 선택 실패(후보 학습 실패). sklearn/torch 설치 상태를 확인하세요.")
        chosen = best["candidate"]
        fit = best
    else:
        fit = _fit_eval_one(chosen)

    yte64 = yte.astype(np.float64)
    pred_te64 = fit["pred_test"].astype(np.float64)
    mse = float(np.mean((pred_te64 - yte64) ** 2))
    mae = float(np.mean(np.abs(pred_te64 - yte64)))
    mape = _mape(yte64, pred_te64)
    acc = _directional_accuracy(yte64, pred_te64)

    last_close = float(np.asarray(prices_1.sort_index().astype(float).to_numpy())[-1])
    pred_next_ret = float(fit["pred_next_return"])
    pred_next_price = float(last_close * (1.0 + pred_next_ret))

    return {
        "model_type": chosen,
        "model": fit["model"],
        "device": fit.get("device", "cpu"),
        "lookback": int(lookback),
        "train_ratio": float(train_ratio),
        "test_metrics": {
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "dir_acc": acc,
        },
        "y_test": yte64,
        "y_pred_test": pred_te64,
        "pred_next_return": pred_next_ret,
        "pred_next_price": pred_next_price,
        "last_close": last_close,
    }


def build_sklearn_dataset(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    lookback: int,
    mode: str = "regression",
    top_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    각 (종목, 날짜 t) 샘플: X = flatten(zscore된 과거 lookback×2), y = 다음 수익률 또는 분류.
    mode: 'regression' | 'binary'
    """
    close, ret, v = _panels(prices, volume, lookback)
    T, L = ret.shape
    tickers = list(prices.columns)
    X_list, y_list, tick_list = [], [], []

    for j, sym in enumerate(tickers):
        for t in range(lookback, T - 1):
            rw = ret[t - lookback : t, j]
            vw = v[t - lookback : t, j]
            x2 = np.stack([rw, vw], axis=-1)  # (P,2)
            x2 = zscore_timesteps(x2.reshape(1, lookback, 2)).reshape(lookback, 2)
            nxt = ret[t + 1, j]
            if not np.isfinite(nxt):
                continue
            X_list.append(x2.reshape(-1))
            y_list.append(float(nxt))
            tick_list.append(sym)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float64)

    tick_arr = np.array(tick_list, dtype=object)

    if mode == "binary":
        thr_hi = np.quantile(y, 1.0 - top_frac)
        thr_lo = np.quantile(y, top_frac)
        y_bin = np.zeros(len(y), dtype=np.int64)
        y_bin[y >= thr_hi] = 1
        y_bin[y <= thr_lo] = 0
        mid = (y > thr_lo) & (y < thr_hi)
        keep = ~mid
        X = X[keep]
        y_bin = y_bin[keep]
        tick_list = tick_arr[keep].tolist()
        return X, y_bin.astype(np.float64), np.ones(len(y_bin)), tick_list

    return X, y, np.ones(len(y), dtype=np.float32), tick_list


def train_predict_sklearn(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    *,
    model_type: str = "ridge",
    lookback: int = 20,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    # 데이터 충분성 체크 및 자동 조정
    close, ret, v = _panels(prices, volume, lookback)
    if len(ret) < lookback + 5:
        lookback = max(5, len(ret) - 5)
        close, ret, v = _panels(prices, volume, lookback)
        print(f"⚠️  lookback을 {lookback}으로 자동 조정 (데이터: {len(ret)}거래일)")

    X, y, _, ticks = build_sklearn_dataset(prices, volume, lookback, mode="regression")
    if len(X) < 20:
        raise ValueError(
            f"학습 샘플이 너무 적습니다 ({len(X)}개).\n"
            f"더 긴 기간·많은 종목의 데이터가 필요합니다."
        )

    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * train_ratio)
    tr, te = idx[:split], idx[split:]

    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    elif model_type == "svm":
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.01))
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    model.fit(X[tr], y[tr])
    pred_te = model.predict(X[te])
    mse = float(np.mean((pred_te - y[te]) ** 2))

    # 최신 시점 각 티커 예측
    close, ret, v = _panels(prices, volume, lookback)
    T, L = ret.shape
    last_scores = {}
    for j, sym in enumerate(prices.columns):
        t = T - 1
        if t < lookback:
            continue
        rw = ret[t - lookback : t, j]
        vw = v[t - lookback : t, j]
        x2 = np.stack([rw, vw], axis=-1)
        x2 = zscore_timesteps(x2.reshape(1, lookback, 2)).reshape(lookback, 2)
        xf = x2.reshape(1, -1)
        if not np.all(np.isfinite(xf)):
            continue
        last_scores[sym] = float(model.predict(xf)[0])

    pred = pd.Series(last_scores, name="pred_return").sort_values(ascending=False)
    return {
        "model": model,
        "model_name": model_type.upper(),
        "test_mse": mse,
        "pred": pred,
        "n_train": len(tr),
        "n_test": len(te),
        "train_metrics": {"loss": 0.0},  # dummy
        "test_metrics": {"loss": mse},
    }


class _SeqRegressor(nn.Module):
    def __init__(self, cell: str, in_dim: int, hidden: int, num_layers: int = 1):
        super().__init__()
        self.cell = cell
        if cell == "lstm":
            self.rnn = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=0.0)
        else:
            self.rnn = nn.GRU(in_dim, hidden, num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def train_predict_rnn(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    *,
    cell: str = "lstm",
    lookback: int = 20,
    hidden: int = 32,
    num_layers: int = 1,
    epochs: int = 15,
    lr: float = 1e-3,
    train_ratio: float = 0.7,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    torch.manual_seed(seed)
    close, ret, v = _panels(prices, volume, lookback)
    T, L = close.shape
    tickers = list(prices.columns)

    # 데이터 충분성 체크 및 자동 조정
    if T < lookback + 5:
        lookback = max(5, T - 5)
        close, ret, v = _panels(prices, volume, lookback)
        T, L = close.shape
        print(f"⚠️  lookback을 {lookback}으로 자동 조정 (데이터: {T}거래일)")

    Xs, ys = [], []
    for j in range(L):
        for t in range(lookback, T - 1):
            rw = ret[t - lookback : t, j]
            vw = v[t - lookback : t, j]
            x2 = np.stack([rw, vw], axis=-1).astype(np.float32)
            x2 = zscore_timesteps(x2.reshape(1, lookback, 2)).reshape(lookback, 2)
            nxt = ret[t + 1, j]
            if not np.isfinite(nxt):
                continue
            Xs.append(x2)
            ys.append(float(nxt))

    if len(Xs) < 20:
        raise ValueError(
            f"RNN 학습 샘플이 너무 적습니다 ({len(Xs)}개).\n"
            f"더 긴 기간·많은 종목의 데이터가 필요합니다."
        )

    X_arr = np.stack(Xs, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)
    n = len(X_arr)
    perm = np.random.RandomState(seed).permutation(n)
    split = int(n * train_ratio)
    tr, te = perm[:split], perm[split:]

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _SeqRegressor(cell, in_dim=2, hidden=hidden, num_layers=num_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(X_t[tr].to(dev))
        loss = nn.functional.mse_loss(pred, y_t[tr].to(dev))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        test_mse = float(
            nn.functional.mse_loss(model(X_t[te].to(dev)), y_t[te].to(dev)).cpu().item()
        )

    last_scores = {}
    t = T - 1
    for j, sym in enumerate(tickers):
        if t < lookback:
            break
        rw = ret[t - lookback : t, j]
        vw = v[t - lookback : t, j]
        x2 = np.stack([rw, vw], axis=-1).astype(np.float32)
        x2 = zscore_timesteps(x2.reshape(1, lookback, 2)).reshape(1, lookback, 2)
        xt = torch.tensor(x2, dtype=torch.float32, device=dev)
        with torch.no_grad():
            last_scores[sym] = float(model(xt).cpu().item())

    pred = pd.Series(last_scores, name="pred_return").sort_values(ascending=False)
    return {
        "model": model,
        "model_name": cell.upper(),
        "test_mse": test_mse,
        "pred": pred,
        "device": str(dev),
        "train_metrics": {"loss": float(np.mean([0.0]))},  # 기록을 위한 dummy
        "test_metrics": {"loss": test_mse},
    }


def train_predict_linear_return(
    prices: pd.DataFrame,
    *,
    lookback: int = 5,
) -> dict:
    """단순 선형 모멘텀·평균회귀 베이스라인: 최근 lookback 누적수익률로 랭킹 (교육용)."""
    prices = prices.sort_index().astype(float)
    r = prices.pct_change()
    mom = r.iloc[-lookback:].sum()
    pred = mom.sort_values(ascending=False)
    pred.name = "momentum_sum"
    return {
        "model": None,
        "model_name": "LinearMomentum",
        "test_mse": float("nan"),
        "pred": pred,
    }
