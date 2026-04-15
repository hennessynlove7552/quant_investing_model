"""
QuantFormer (arXiv:2404.00424v3) — 논문 구조에 맞춘 근사 구현.

핵심 (논문 §3):
- 수치 시계열: 각 시점 행 벡터 x = [일(또는 기간) 수익률 r, 회전율(turnover) v]
- 시점별로 횡단면(N종목) Z-score 정규화 (식 4)
- word embedding 대신 선형 임베딩 (식 7), 위치 인코딩 없음 (시계열 순서는 토큰 순서에 내재)
- 인코더만: multi-head self-attention + FFN, 디코더/마스킹 제거
- 출력: 각 종목별 ϱ차원 softmax (식 11–12)
- 손실: 예측 분포와 원-핫 타깃 간 MSE (식 13)
- 타깃: 다음 기간 수익률의 경험적 분위 CDF Ψ 기반 양자화 라벨 (ϱ=3, φ=0.2 등)

거래량이 없으면 v를 |수익률|/변동성 등으로 대체 (교육용 근사).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _turnover_proxy(
    ret: np.ndarray,
    volume: Optional[np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """일별 turnover rate 근사. volume이 있으면 정규화 거래량, 없으면 변동성 대비 절대수익."""
    T, L = ret.shape
    if volume is not None and np.isfinite(volume).mean() > 0.5:
        vol = np.asarray(volume, dtype=np.float64)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        base = np.maximum(np.nanmean(np.where(vol > 0, vol, np.nan), axis=0), eps)
        v = vol / (base.reshape(1, -1) + eps)
        v = np.clip(v, 0.0, 50.0)
        return v.astype(np.float32)
    # price-only proxy
    sig = np.zeros_like(ret, dtype=np.float64)
    for t in range(T):
        w = ret[max(0, t - 19) : t + 1]
        if w.shape[0] < 2:
            sig[t] = np.nanstd(ret[: max(1, t + 1)], axis=0) + eps
        else:
            sig[t] = np.nanstd(w, axis=0) + eps
    v = np.abs(ret) / sig
    return np.nan_to_num(v, nan=0.0).astype(np.float32)


def zscore_timesteps(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x: (N, P, F) — 각 시점 p에 대해 N종목 횡단면으로 평균·표준편차 (논문 식 4).
    """
    out = x.astype(np.float64).copy()
    N, P, F = out.shape
    for p in range(P):
        for f in range(F):
            col = out[:, p, f]
            mu = np.nanmean(col)
            sig = np.nanstd(col) + eps
            out[:, p, f] = (col - mu) / sig
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def quantile_labels(
    next_ret: np.ndarray,
    rho: int = 3,
    phi: float = 0.2,
    include_null: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    다음 기간 수익률 next_ret: (N,)
    반환: y (N, rho), mask (N,) — mask=1인 샘플만 학습.
    rho=3, phi=0.2: 상/중/하 각 20% 구간만 원-핫, 가운데 공백은 0벡터 (논문 §3.2).
    
    단일 종목(N=1)은 특수 처리: 수익률 부호로 라벨링
    """
    N = len(next_ret)
    r = np.asarray(next_ret, dtype=np.float64)
    valid = np.isfinite(r)
    
    y = np.zeros((N, rho), dtype=np.float32)
    m = np.zeros(N, dtype=np.float32)

    # 단일 종목: 수익률 부호로 분류
    if N == 1:
        if valid[0]:
            if r[0] > 0:
                y[0, 0] = 1.0  # 상승
            elif r[0] < 0:
                y[0, 2] = 1.0  # 하락
            else:
                y[0, 1] = 1.0  # 중립
            m[0] = 1.0
        return y, m
    
    order = np.argsort(r)
    ranks = np.empty(N, dtype=np.float64)
    ranks[order] = np.arange(N, dtype=np.float64)
    psi = ranks / max(N - 1, 1)

    k = max(1, int(round(phi * N)))

    if rho == 3 and include_null:
        # 하위 k → class 2, 상위 k → class 0, 나머지에서 중간 k → class 1 (겹침 제거)
        bottom_idx = set(order[:k].tolist())
        top_idx = set(order[-k:].tolist())
        rest = [i for i in order.tolist() if i not in bottom_idx and i not in top_idx]
        mid_idx = rest[:k] if len(rest) >= k else rest

        for i in top_idx:
            y[i, 0] = 1.0
            m[i] = 1.0
        for i in mid_idx:
            y[i, 1] = 1.0
            m[i] = 1.0
        for i in bottom_idx:
            y[i, 2] = 1.0
            m[i] = 1.0
    elif rho == 5 and not include_null:
        for c in range(5):
            lo = int(N * (c * 0.2))
            hi = int(N * ((c + 1) * 0.2))
            if hi <= lo:
                continue
            idx = order[lo:hi]
            y[idx, c] = 1.0
            m[idx] = 1.0
    else:
        # 일반: psi 구간으로 균등 분할 (간이)
        for c in range(rho):
            lo = int(N * (c / rho))
            hi = int(N * ((c + 1) / rho))
            if hi <= lo:
                continue
            idx = order[lo:hi]
            y[idx, c] = 1.0
            m[idx] = 1.0

    m = m * valid.astype(np.float32)
    return y, m


@dataclass
class QFDayBatch:
    x: torch.Tensor  # (N, P, 2)
    y: torch.Tensor  # (N, rho)
    mask: torch.Tensor  # (N,)


class QuantFormer(nn.Module):
    """
    논문: 선형 임베딩 → L층 TransformerEncoder (positional encoding 없음) → 마지막 시점 풀링 → ϱ-way softmax.
    """

    def __init__(
        self,
        in_dim: int = 2,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 64,
        rho: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, rho)
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, 2)
        h = self.embed(x)
        h = self.encoder(h)
        last = h[:, -1, :]
        logits = self.head(last)
        return F.softmax(logits, dim=-1)


def build_daily_batches(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    seq_len: int = 20,
    rho: int = 3,
    phi: float = 0.2,
    include_null: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    close, volume: (T, L). volume optional.
    각 거래일 t에 대해 입력은 t-seq_len..t-1, 라벨은 t→t+1 수익률.
    """
    T, L = close.shape
    if T < seq_len + 2:
        return []

    eps = 1e-12
    ret = np.zeros((T, L), dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps)

    v = _turnover_proxy(ret.astype(np.float32), volume)

    batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(seq_len, T - 1):
        r_win = ret[t - seq_len : t, :]  # (P, L)
        v_win = v[t - seq_len : t, :]
        x = np.stack([r_win.T, v_win.T], axis=-1)  # (L, P, 2)
        x = zscore_timesteps(x)

        next_r = ret[t + 1]
        y, mask = quantile_labels(next_r, rho=rho, phi=phi, include_null=include_null)
        
        # 단일 종목일 경우 마스크 길이 조정
        # L=1일 때는 최소 1개 레이블만 필요
        min_mask_threshold = 1 if L == 1 else 3
        if mask.sum() < min_mask_threshold:
            continue
        batches.append((x.astype(np.float32), y, mask))

    return batches


def train_quantformer(
    batches: List[QFDayBatch],
    *,
    device: torch.device,
    rho: int = 3,
    d_model: int = 16,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[QuantFormer, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QuantFormer(
        in_dim=2,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=max(64, d_model * 4),
        rho=rho,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"loss": []}
    for _ in range(epochs):
        model.train()
        day_losses = []
        for b in batches:
            x = b.x.to(device)
            y = b.y.to(device)
            m = b.mask.to(device)
            if m.sum() < 1:
                continue
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred[m > 0], y[m > 0])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            day_losses.append(loss.item())
        history["loss"].append(float(np.mean(day_losses)) if day_losses else float("nan"))

    return model, history


@torch.no_grad()
def evaluate_qf(model: QuantFormer, batches: List[QFDayBatch], device: torch.device) -> dict:
    model.eval()
    losses = []
    accs = []
    for b in batches:
        x = b.x.to(device)
        y = b.y.to(device)
        m = b.mask.to(device)
        if m.sum() < 1:
            continue
        pred = model(x)
        loss = F.mse_loss(pred[m > 0], y[m > 0])
        losses.append(loss.item())
        pred_c = pred[m > 0].argmax(dim=-1)
        true_c = y[m > 0].argmax(dim=-1)
        accs.append((pred_c == true_c).float().mean().item())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
        "n": len(batches),
    }


def train_and_predict_quantformer(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    *,
    seq_len: int = 20,
    rho: int = 3,
    phi: float = 0.2,
    include_null: bool = True,
    train_ratio: float = 0.7,
    epochs: int = 20,
    lr: float = 1e-3,
    d_model: int = 16,
    nhead: int = 4,
    num_layers: int = 2,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    prices = prices.sort_index().astype(float)
    close = prices.to_numpy()
    vol_arr = None
    if volume is not None and not volume.empty:
        vol_aligned = volume.reindex(prices.index).reindex(columns=prices.columns)
        vol_arr = vol_aligned.astype(float).to_numpy()

    raw = build_daily_batches(
        close,
        vol_arr,
        seq_len=seq_len,
        rho=rho,
        phi=phi,
        include_null=include_null,
    )
    
    # 배치 생성 실패 시 seq_len 자동 축소
    if not raw:
        # seq_len을 선형적으로 축소하면서 재시도
        for reduced_seq_len in range(max(5, seq_len // 2), 4, -1):
            raw = build_daily_batches(
                close,
                vol_arr,
                seq_len=reduced_seq_len,
                rho=rho,
                phi=phi,
                include_null=include_null,
            )
            if raw:
                print(f"⚠️  seq_len을 {seq_len} → {reduced_seq_len}로 자동 조정")
                seq_len = reduced_seq_len
                break
    
    if not raw:
        T, L = close.shape
        raise ValueError(
            f"QuantFormer용 배치를 생성할 수 없습니다.\n"
            f"  • 데이터: {T}거래일, {L}종목\n"
            f"  • 필요: 최소 {26}거래일 이상\n"
            f"  • 권장: {100}거래일 이상\n"
            f"더 긴 기간의 데이터를 사용해주세요."
        )

    tickers = list(prices.columns)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batches = [
        QFDayBatch(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.float32),
        )
        for x, y, mask in raw
    ]

    n = len(batches)
    split = int(max(1, min(n - 1, round(n * train_ratio))))
    train_b, test_b = batches[:split], batches[split:]

    model, hist = train_quantformer(
        train_b,
        device=dev,
        rho=rho,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )

    train_m = evaluate_qf(model, train_b, dev)
    test_m = evaluate_qf(model, test_b, dev) if test_b else {"loss": np.nan, "acc": np.nan, "n": 0}

    last = batches[-1]
    model.eval()
    with torch.no_grad():
        p = model(last.x.to(dev)).cpu().numpy()

    # 상승(상위 분위) 확률을 랭킹 점수로 사용: rho=3에서 class 0 = top bucket
    if rho == 3:
        score = p[:, 0] - p[:, 2]
    else:
        score = p[:, 0] - p[:, -1]

    pred = pd.Series(score, index=tickers, name="score").sort_values(ascending=False)
    return {
        "model": model,
        "model_name": "QuantFormer",
        "history": hist,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "pred": pred,
        "proba": pd.DataFrame(p, index=tickers, columns=[f"P(class_{i})" for i in range(rho)]),
        "rho": rho,
        "device": str(dev),
    }
