"""
THGNN (Temporal & Heterogeneous GNN) - 논문 로직을 단순화해 구현.

원 논문(Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction, CIKM'22)의
핵심 아이디어를 현재 레포(종가 중심 데이터) 환경에 맞춰 반영합니다.

- 일자별로 과거 윈도우의 상관계수로 동적 관계 그래프를 생성
  - 양(positive) / 음(negative) 관계 2종(edge type)
- 각 종목의 과거 P일 특징 시퀀스를 Transformer encoder로 임베딩
- 관계 그래프에서 (양/음) GAT로 이웃 메시지 집계
- self/pos/neg 3개 메시지 소스를 이종(heterogeneous) 어텐션으로 결합
- 다음날 수익률 랭킹 기반 top-k / bottom-k만 라벨링한 semi-supervised node classification(BCE)

주의:
- 이 구현은 "논문 로직을 고려"한 실용적 근사입니다.
- 입력이 종가만 있을 때도 동작하도록 6차원 특징을 종가에서 생성합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Transformer용 사인/코사인 positional encoding. shape=(length, dim)"""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_features_from_close(close: np.ndarray) -> np.ndarray:
    """
    종가(close)만으로 6차원 일별 특징 생성.

    입력
    - close: shape (T, L)  (T=날짜, L=종목수)

    출력
    - feat: shape (T, L, 6)
    """
    close = np.asarray(close, dtype=np.float64)
    if close.ndim != 2:
        raise ValueError("close는 (T, L) 2차원 배열이어야 합니다.")

    T, L = close.shape
    eps = 1e-12

    ret = np.zeros((T, L), dtype=np.float64)
    ret[1:] = close[1:] / (close[:-1] + eps) - 1.0
    logret = np.zeros((T, L), dtype=np.float64)
    logret[1:] = np.log((close[1:] + eps) / (close[:-1] + eps))

    def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.full_like(x, np.nan, dtype=np.float64)
        sig = np.full_like(x, np.nan, dtype=np.float64)
        for t in range(w - 1, T):
            window = x[t - w + 1 : t + 1]
            mu[t] = np.nanmean(window, axis=0)
            sig[t] = np.nanstd(window, axis=0)
        return mu, sig

    mu5, sd5 = _rolling_mean_std(ret, 5)
    mu20, sd20 = _rolling_mean_std(ret, 20)

    z5 = (ret - mu5) / (sd5 + eps)
    z20 = (ret - mu20) / (sd20 + eps)

    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.zeros_like(x, dtype=np.float64)
        out[0] = x[0]
        for t in range(1, T):
            out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
        return out

    ema5 = _ema(close, 5)
    ema20 = _ema(close, 20)
    ema5_diff = (ema5 - close) / (close + eps)
    ema20_diff = (ema20 - close) / (close + eps)

    feat = np.stack([ret, logret, z5, z20, ema5_diff, ema20_diff], axis=-1)  # (T, L, 6)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)


def correlation_graph(
    returns_window: np.ndarray,
    threshold: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    과거 윈도우(예: 20일)의 수익률로 상관계수 행렬을 만들고,
    양/음 관계 2종 adjacency 생성.

    입력
    - returns_window: shape (W, L)

    출력
    - A_pos, A_neg: shape (L, L), {0,1} (자기자신 제외)
    """
    W, L = returns_window.shape
    if W < 2:
        raise ValueError("상관 그래프 생성을 위해 윈도우 길이는 최소 2 이상이어야 합니다.")

    x = returns_window.astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    A_pos = (corr > threshold).astype(np.float32)
    A_neg = (corr < -threshold).astype(np.float32)
    np.fill_diagonal(A_pos, 0.0)
    np.fill_diagonal(A_neg, 0.0)
    return A_pos, A_neg


def make_labels_ranked(
    next_returns: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    다음날 수익률 랭킹으로 top-k=1, bottom-k=0 라벨 생성.
    나머지는 마스크로 학습/평가에서 제외(semi-supervised).

    입력
    - next_returns: shape (L,)
    - top_k: int (>=1)

    출력
    - y: shape (L,) in {0,1}
    - mask: shape (L,) in {0,1}
    """
    r = np.asarray(next_returns, dtype=np.float64)
    L = r.shape[0]
    k = int(top_k)
    k = max(1, min(k, L // 2))

    order = np.argsort(r)  # ascending
    bottom = order[:k]
    top = order[-k:]

    y = np.zeros((L,), dtype=np.float32)
    y[top] = 1.0
    y[bottom] = 0.0

    mask = np.zeros((L,), dtype=np.float32)
    mask[top] = 1.0
    mask[bottom] = 1.0
    return y, mask


@dataclass
class THGNNBatch:
    """
    하루(t) 단위 배치.
    - x: (L, P, F)
    - A_pos/A_neg: (L, L)
    - y/mask: (L,)
    """

    x: torch.Tensor
    A_pos: torch.Tensor
    A_neg: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, model_dim: int = 128, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L, P, F)
        return: (L, D)
        """
        L, P, _ = x.shape
        h = self.in_proj(x)  # (L, P, D)
        pe = _positional_encoding(P, self.model_dim, x.device)  # (P, D)
        h = h + pe.unsqueeze(0)
        h = self.dropout(h)
        h = self.encoder(h)  # (L, P, D)
        return h[:, -1, :]  # 마지막 시점 임베딩 사용


class GATRelation(nn.Module):
    """
    단일 관계(예: positive 또는 negative)용 GAT.
    adjacency가 희소하지만 여기서는 L이 크지 않다는 가정으로 dense mask로 구현.
    """

    def __init__(self, in_dim: int, out_dim: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads
        if out_dim % n_heads != 0:
            raise ValueError("out_dim은 n_heads로 나누어 떨어져야 합니다.")

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        h: (L, D_in)
        A: (L, L) in {0,1}
        return: (L, D_out)
        """
        L, _ = h.shape
        Wh = self.W(h)  # (L, out_dim)
        Wh = Wh.view(L, self.n_heads, self.head_dim)  # (L, H, Hd)

        # e_ij 계산: a^T [Wh_i || Wh_j]
        # dense로 만들되 adjacency로 마스킹
        Wh_i = Wh.unsqueeze(1)  # (L, 1, H, Hd)
        Wh_j = Wh.unsqueeze(0)  # (1, L, H, Hd)
        cat = torch.cat([Wh_i.expand(L, L, -1, -1), Wh_j.expand(L, L, -1, -1)], dim=-1)  # (L, L, H, 2Hd)
        e = (cat * self.attn.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (L, L, H)
        e = self.leaky_relu(e)

        # 마스크: adjacency==0은 -inf 처리해서 softmax에서 제외
        mask = (A > 0).unsqueeze(-1)  # (L, L, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=1)  # (L, L, H)  i 기준으로 neighbor j에 softmax
        # 이웃이 하나도 없는 노드는 e가 전부 -inf가 되어 softmax가 NaN이 될 수 있음
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.dropout(alpha)

        # 집계: sum_j alpha_ij * Wh_j
        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)  # (L, H, Hd)
        out = out.reshape(L, self.out_dim)
        out = self.out_proj(out)
        return out


class HeteroAggregator(nn.Module):
    """
    self/pos/neg 3개 메시지 소스를 이종 어텐션으로 결합.
    논문 식(8)-(9)의 global importance(노드 평균) 방식을 따름.
    """

    def __init__(self, dim: int, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp_self = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh())
        self.mlp_pos = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh())
        self.mlp_neg = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh())
        self.q = nn.Parameter(torch.empty(attn_dim))
        nn.init.normal_(self.q, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def _score(self, x: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        # x: (L, D) -> (L, A) -> (L,) -> 평균
        h = mlp(x)  # (L, A)
        s = torch.matmul(h, self.q)  # (L,)
        return s.mean()  # scalar

    def forward(self, h_self: torch.Tensor, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
        k_self = self._score(h_self, self.mlp_self)
        k_pos = self._score(h_pos, self.mlp_pos)
        k_neg = self._score(h_neg, self.mlp_neg)
        w = torch.softmax(torch.stack([k_self, k_pos, k_neg], dim=0), dim=0)  # (3,)
        out = w[0] * h_self + w[1] * h_pos + w[2] * h_neg
        return self.dropout(out)


class THGNN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 6,
        enc_dim: int = 128,
        enc_heads: int = 8,
        enc_layers: int = 2,
        gat_dim: int = 256,
        gat_heads: int = 4,
        hetero_attn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(feature_dim, model_dim=enc_dim, n_heads=enc_heads, n_layers=enc_layers, dropout=dropout)

        # self 메시지 선형 변환 (논문: W_self * H_enc)
        self.self_proj = nn.Linear(enc_dim, gat_dim)

        self.gat_pos = GATRelation(enc_dim, out_dim=gat_dim, n_heads=gat_heads, dropout=dropout)
        self.gat_neg = GATRelation(enc_dim, out_dim=gat_dim, n_heads=gat_heads, dropout=dropout)

        self.hetero = HeteroAggregator(gat_dim, attn_dim=hetero_attn_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(gat_dim, gat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_dim, 1),
        )

    def forward(self, x: torch.Tensor, A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        """
        x: (L, P, F)
        A_pos/A_neg: (L, L)
        return: prob (L,) in [0,1]
        """
        h_enc = self.encoder(x)  # (L, enc_dim)
        h_self = self.self_proj(h_enc)  # (L, gat_dim)
        h_pos = self.gat_pos(h_enc, A_pos)  # (L, gat_dim)
        h_neg = self.gat_neg(h_enc, A_neg)  # (L, gat_dim)
        z = self.hetero(h_self, h_pos, h_neg)  # (L, gat_dim)
        logits = self.classifier(z).squeeze(-1)  # (L,)
        return torch.sigmoid(logits)


def make_daily_batches(
    features: np.ndarray,
    close: np.ndarray,
    lookback: int = 20,
    corr_window: int = 20,
    corr_threshold: float = 0.6,
    top_k: int = 10,
) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    features: (T, L, F)
    close: (T, L)

    return list of (x, A_pos, A_neg, y, mask) for day t.
    day t는 x가 (t-lookback ... t-1) 구간을 사용하고, 라벨은 t->t+1 수익률로 생성.
    """
    T, L, Fdim = features.shape
    if lookback < 2:
        raise ValueError("lookback은 최소 2 이상이어야 합니다.")
    if corr_window < 2:
        raise ValueError("corr_window는 최소 2 이상이어야 합니다.")
    if T <= lookback + 1:
        raise ValueError("데이터 길이가 너무 짧습니다. lookback/기간을 늘려주세요.")

    # 수익률(그래프 생성/라벨용)
    eps = 1e-12
    ret = np.zeros((T, L), dtype=np.float32)
    ret[1:] = (close[1:] / (close[:-1] + eps) - 1.0).astype(np.float32)

    batches = []
    # t는 "예측을 수행할 거래일"로 보고, 입력은 t 이전 lookback일.
    # 라벨은 t 다음날(t+1) 수익률 랭킹으로 생성.
    for t in range(lookback, T - 1):
        x = features[t - lookback : t]  # (P, L, F) -> transpose to (L, P, F)
        x = np.transpose(x, (1, 0, 2)).astype(np.float32)

        # corr graph는 과거 corr_window일 수익률로 생성 (논문: past 20 trading days)
        w_start = max(0, t - corr_window)
        returns_window = ret[w_start:t]  # (W, L)
        if returns_window.shape[0] < 2:
            continue
        A_pos, A_neg = correlation_graph(returns_window, threshold=corr_threshold)

        next_r = ret[t + 1]  # (L,)
        y, mask = make_labels_ranked(next_r, top_k=top_k)

        batches.append((x, A_pos, A_neg, y, mask))

    return batches


@torch.no_grad()
def evaluate_batches(
    model: THGNN,
    batches: list[THGNNBatch],
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    accs = []
    for b in batches:
        p = model(b.x.to(device), b.A_pos.to(device), b.A_neg.to(device))
        y = b.y.to(device)
        m = b.mask.to(device)
        if m.sum() < 1:
            continue
        loss = F.binary_cross_entropy(p[m > 0], y[m > 0])
        pred = (p[m > 0] >= 0.5).float()
        acc = (pred == y[m > 0]).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
        "n_days": int(len(batches)),
    }


def train_thgnn(
    model: THGNN,
    train_batches: list[THGNNBatch],
    valid_batches: Optional[list[THGNNBatch]] = None,
    *,
    device: Optional[torch.device] = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 10,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train": [], "valid": []}
    for _ in range(int(epochs)):
        model.train()
        day_losses = []
        day_accs = []

        for b in train_batches:
            x = b.x.to(device)
            A_pos = b.A_pos.to(device)
            A_neg = b.A_neg.to(device)
            y = b.y.to(device)
            m = b.mask.to(device)

            if m.sum() < 1:
                continue

            opt.zero_grad(set_to_none=True)
            p = model(x, A_pos, A_neg)
            loss = F.binary_cross_entropy(p[m > 0], y[m > 0])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                pred = (p[m > 0] >= 0.5).float()
                acc = (pred == y[m > 0]).float().mean()
            day_losses.append(loss.item())
            day_accs.append(acc.item())

        train_metrics = {
            "loss": float(np.mean(day_losses)) if day_losses else float("nan"),
            "acc": float(np.mean(day_accs)) if day_accs else float("nan"),
        }
        history["train"].append(train_metrics)

        if valid_batches is not None:
            valid_metrics = evaluate_batches(model, valid_batches, device)
            history["valid"].append(valid_metrics)

    return history

