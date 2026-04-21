"""
논문(2209.10771v3) §3.3–3.4의 **변동성 표면(20×20) 시계열 예측**을 위한 PyTorch 구현 모음.

포함 모델:
- ConvLSTM (Shi et al., 2015) — 식 (6)–(8)
- SA-ConvLSTM — 식 (9)–(16)
- ConvTF — §3.3 (MultiConvAttn 포함 최소 구현)
- PI-ConvTF — §3.4 (Black–Scholes 기반 physics loss)

## 공통 입출력
- 입력 `x_seq`: (B, T, C, H, W)
  - ConvLSTM/SA-ConvLSTM/ConvTF 기본: C=1
  - PI-ConvTF: C=5, 채널 순서는 `[tau, sigma, S_norm, r, K_norm]`
- 출력: (B, 1, H, W)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, np.ndarray]


def _as_tensor(x: TensorLike, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    t = torch.tensor(x)
    return t.to(device) if device is not None else t


def _check_5d(x_seq: torch.Tensor, name: str = "x_seq") -> None:
    if x_seq.ndim != 5:
        raise ValueError(f"{name} must be 5D (B,T,C,H,W). got shape={tuple(x_seq.shape)}")


def l1_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (pred - true).abs().mean()


def mape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = true.abs().clamp_min(eps)
    return ((pred - true).abs() / denom).mean()


# =============================================================================
# Black–Scholes (논문 식 (22)–(23)) + PI-ConvTF physics loss
# =============================================================================


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    S: torch.Tensor,
    K: torch.Tensor,
    r: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    tau = tau.clamp_min(eps)
    sigma = sigma.clamp_min(eps)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S.clamp_min(eps) / K.clamp_min(eps)) + (r + 0.5 * sigma**2) * tau) / (
        sigma * sqrt_tau
    )
    d2 = d1 - sigma * sqrt_tau
    return S * norm_cdf(d1) - K * torch.exp(-r * tau) * norm_cdf(d2)


def make_surface_batches(
    series: TensorLike,
    *,
    lookback: int = 10,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    (T,C,H,W) 시계열 텐서를 supervised pair list로 변환.
    - x_seq: (1,lookback,C,H,W)
    - y_next: (1,1,H,W)  (channel 0을 타깃으로)
    """
    s = _as_tensor(series, device=device).float()
    if s.ndim != 4:
        raise ValueError(f"series must be 4D (T,C,H,W). got shape={tuple(s.shape)}")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    T, C, H, W = s.shape
    if T <= lookback:
        return []

    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for t in range(lookback, T, stride):
        x = s[t - lookback : t].unsqueeze(0)  # (1,lookback,C,H,W)
        y = s[t, 0:1].unsqueeze(0)  # (1,1,H,W)
        out.append((x, y))
    return out


@dataclass
class SurfaceBatch:
    x_seq: torch.Tensor  # (B,T,C,H,W)
    y_next: torch.Tensor  # (B,1,H,W)
    market_next: Optional[Dict[str, torch.Tensor]] = None  # for PI-ConvTF


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (논문 식 (6)–(8))."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv_f = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_g = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_o = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(
        self,
        x_t: torch.Tensor,  # (B,C,H,W)
        h_prev: torch.Tensor,  # (B,Hc,H,W)
        c_prev: torch.Tensor,  # (B,Hc,H,W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xh = torch.cat([x_t, h_prev], dim=1)
        f_t = torch.sigmoid(self.conv_f(xh))
        i_t = torch.sigmoid(self.conv_i(xh))
        g_t = torch.tanh(self.conv_g(xh))
        c_t = f_t * c_prev + i_t * g_t
        o_t = torch.sigmoid(self.conv_o(xh))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.
    Input:  (B,T,C,H,W)
    Output: (B,1,H,W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=kernel_size)
                for i in range(self.num_layers)
            ]
        )
        self.final = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _check_5d(x_seq)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        h = [torch.zeros(B, self.cells[i].hidden_channels, H, W, device=device, dtype=dtype) for i in range(self.num_layers)]
        c = [torch.zeros_like(h_i) for h_i in h]

        for t in range(T):
            x_t = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, h[i], c[i])
                x_t = h[i]
        return self.final(h[-1])


# =============================================================================
# SA-ConvLSTM (논문 식 (9)–(16))
# =============================================================================


class SelfAttentionMemory(nn.Module):
    def __init__(self, channels: int, qk_channels: int = 8):
        super().__init__()
        self.q = nn.Conv2d(channels, qk_channels, kernel_size=1)
        self.kh = nn.Conv2d(channels, qk_channels, kernel_size=1)
        self.vh = nn.Conv2d(channels, channels, kernel_size=1)
        self.km = nn.Conv2d(channels, qk_channels, kernel_size=1)
        self.vm = nn.Conv2d(channels, channels, kernel_size=1)

        self.z = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.o = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.g = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.i = nn.Conv2d(channels * 2, channels, kernel_size=1)

    @staticmethod
    def _flat_hw(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2)  # (B,C,HW)

    @staticmethod
    def _unflat_hw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], H, W)

    def forward(self, H_in: torch.Tensor, M_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = H_in.shape
        Q = self._flat_hw(self.q(H_in))  # (B,qk,HW)
        Kh = self._flat_hw(self.kh(H_in))
        Vh = self._flat_hw(self.vh(H_in))  # (B,C,HW)
        Km = self._flat_hw(self.km(M_prev))
        Vm = self._flat_hw(self.vm(M_prev))

        Q_t = Q.transpose(1, 2)  # (B,HW,qk)
        scale = math.sqrt(Q.shape[1])
        Ah = torch.softmax(torch.bmm(Q_t, Kh) / scale, dim=-1)  # (B,HW,HW)
        Am = torch.softmax(torch.bmm(Q_t, Km) / scale, dim=-1)

        Zh = torch.bmm(Vh, Ah.transpose(1, 2))  # (B,C,HW)
        Zm = torch.bmm(Vm, Am.transpose(1, 2))
        Zh = self._unflat_hw(Zh, H, W)
        Zm = self._unflat_hw(Zm, H, W)
        Z = self.z(torch.cat([Zh, Zm], dim=1))

        concat = torch.cat([H_in, Z], dim=1)
        O_t = torch.sigmoid(self.o(concat))
        G_t = torch.tanh(self.g(concat))
        I_t = torch.sigmoid(self.i(concat))
        M_t = (1.0 - I_t) * M_prev + I_t * G_t
        H_out = O_t * M_t
        return H_out, M_t


class SAConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, qk_channels: int = 8):
        super().__init__()
        self.base = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)
        self.sam = SelfAttentionMemory(hidden_channels, qk_channels=qk_channels)
        self.hidden_channels = hidden_channels

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, m_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_in, c_t = self.base(x_t, h_prev, c_prev)
        h_out, m_t = self.sam(h_in, m_prev)
        return h_out, c_t, m_t


class SAConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        qk_channels: int = 8,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.cells = nn.ModuleList(
            [
                SAConvLSTMCell(in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size, qk_channels)
                for i in range(self.num_layers)
            ]
        )
        self.final = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _check_5d(x_seq)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        h = [torch.zeros(B, self.cells[i].hidden_channels, H, W, device=device, dtype=dtype) for i in range(self.num_layers)]
        c = [torch.zeros_like(h_i) for h_i in h]
        m = [torch.zeros_like(h_i) for h_i in h]

        for t in range(T):
            x_t = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i], m[i] = cell(x_t, h[i], c[i], m[i])
                x_t = h[i]
        return self.final(h[-1])


# =============================================================================
# ConvTF (논문 §3.3, 식 (17)–(21)) - minimal implementation
# =============================================================================


class MultiConvAttn(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, kernel_size: int = 3):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by heads({heads})")
        self.d_model = d_model
        self.heads = heads
        self.dh = d_model // heads
        padding = kernel_size // 2

        self.W1 = nn.ModuleList([nn.Conv2d(d_model, self.dh, kernel_size, padding=padding) for _ in range(heads)])
        self.W2 = nn.ModuleList([nn.Conv2d(d_model, self.dh, kernel_size, padding=padding) for _ in range(heads)])
        self.W3 = nn.ModuleList([nn.Conv2d(self.dh * 2, 1, kernel_size, padding=padding) for _ in range(heads)])
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _check_5d(seq, "seq")
        B, T, D, H, W = seq.shape
        outs: List[torch.Tensor] = []
        for k in range(T):
            Ik = seq[:, k]
            head_outs: List[torch.Tensor] = []
            for j in range(self.heads):
                Qk = self.W1[j](Ik)
                scores: List[torch.Tensor] = []
                values: List[torch.Tensor] = []
                for i in range(T):
                    Ii = seq[:, i]
                    Ki = self.W2[j](Ii)
                    Vi = Ki
                    Hij = self.W3[j](torch.cat([Qk, Ki], dim=1))
                    scores.append(Hij)
                    values.append(Vi)
                score_stack = torch.stack(scores, dim=1)  # (B,T,1,H,W)
                A = torch.softmax(score_stack, dim=1)
                V_stack = torch.stack(values, dim=1)  # (B,T,dh,H,W)
                head_outs.append((A * V_stack).sum(dim=1))
            Ok = torch.cat(head_outs, dim=1)  # (B,D,H,W)
            outs.append(self.proj(Ok))
        return torch.stack(outs, dim=1)


class ConvTFBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiConvAttn(d_model=d_model, heads=heads)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.GroupNorm(1, d_model)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.ff = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq + self.drop(self.attn(seq))
        B, T, D, H, W = seq.shape
        x = self.norm1(seq.view(B * T, D, H, W))
        x = x + self.drop(self.ff(x))
        x = self.norm2(x)
        return x.view(B, T, D, H, W)


class ConvTF(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 32,
        heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        sffn_depth: int = 10,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([ConvTFBlock(d_model=d_model, heads=heads, dropout=dropout) for _ in range(int(num_layers))])

        sffn: List[nn.Module] = []
        for _ in range(max(1, int(sffn_depth))):
            sffn.append(nn.Conv2d(d_model, d_model, kernel_size=3, padding=1))
            sffn.append(nn.LeakyReLU(0.2, inplace=True))
        sffn.append(nn.Conv2d(d_model, 1, kernel_size=1))
        self.sffn = nn.Sequential(*sffn)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _check_5d(x_seq)
        B, T, C, H, W = x_seq.shape
        x = self.embed(x_seq.view(B * T, C, H, W)).view(B, T, -1, H, W)
        for blk in self.blocks:
            x = blk(x)
        return self.sffn(x[:, -1])  # last token (query-free minimal decoder)


# =============================================================================
# PI-ConvTF (논문 §3.4)
# =============================================================================


@dataclass
class PIConvTFLoss:
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    total_loss: torch.Tensor


class PIConvTF(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        heads: int = 4,
        num_layers: int = 1,
        lambda_phys: float = 0.1,
        dropout: float = 0.0,
        sffn_depth: int = 10,
    ):
        super().__init__()
        self.backbone = ConvTF(
            in_channels=5, d_model=d_model, heads=heads, num_layers=num_layers, dropout=dropout, sffn_depth=sffn_depth
        )
        self.lambda_phys = float(lambda_phys)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.backbone(x_seq)

    def compute_loss(
        self,
        x_seq: torch.Tensor,
        sigma_true_next: torch.Tensor,
        *,
        market_next: Dict[str, torch.Tensor],
        loss_type: str = "l1",
        physics_norm: str = "l1",
        eps: float = 1e-12,
    ) -> PIConvTFLoss:
        sigma_pred = self.forward(x_seq)
        if loss_type == "mape":
            data = mape(sigma_pred, sigma_true_next, eps=eps)
        elif loss_type == "mse":
            data = F.mse_loss(sigma_pred, sigma_true_next)
        else:
            data = l1_loss(sigma_pred, sigma_true_next)

        tau = market_next["tau"]
        S = market_next["S"]
        r = market_next["r"]
        K = market_next["K"]

        # NOTE:
        # `detach()` + later `.to(device)` on the *same* tensor object can be problematic for leaf tensors
        # (especially when moving to CUDA). For PINN-style autograd we only need fresh leaf variables
        # with requires_grad=True on the correct device/dtype.
        tau_req = tau.to(dtype=sigma_pred.dtype).requires_grad_(True)
        S_req = S.to(dtype=sigma_pred.dtype).requires_grad_(True)
        C_eval = black_scholes_call(S_req, K, r, tau_req, sigma_pred.clamp_min(eps), eps=eps)

        dC_dtau = torch.autograd.grad(
            C_eval, tau_req, grad_outputs=torch.ones_like(C_eval), create_graph=True, retain_graph=True
        )[0]
        dC_dS = torch.autograd.grad(
            C_eval, S_req, grad_outputs=torch.ones_like(C_eval), create_graph=True, retain_graph=True
        )[0]
        d2C_dS2 = torch.autograd.grad(
            dC_dS, S_req, grad_outputs=torch.ones_like(dC_dS), create_graph=True, retain_graph=True
        )[0]

        residual = dC_dtau - r * C_eval + r * S_req * dC_dS + 0.5 * (sigma_pred**2) * (S_req**2) * d2C_dS2
        phys = (residual**2).mean() if physics_norm == "l2" else residual.abs().mean()
        total = data + self.lambda_phys * phys
        return PIConvTFLoss(data_loss=data, physics_loss=phys, total_loss=total)


def train_model(
    model: nn.Module,
    train_batches: Sequence[SurfaceBatch],
    *,
    device: Optional[str] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
) -> List[float]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []

    for _ in range(int(epochs)):
        model.train()
        losses: List[float] = []
        for b in train_batches:
            x = b.x_seq.to(dev)
            y = b.y_next.to(dev)
            opt.zero_grad(set_to_none=True)
            if isinstance(model, PIConvTF):
                if b.market_next is None:
                    raise ValueError("PIConvTF training requires SurfaceBatch.market_next")
                mn = {k: v.to(dev) for k, v in b.market_next.items()}
                loss_obj = model.compute_loss(x, y, market_next=mn, loss_type="l1")
                loss = loss_obj.total_loss
            else:
                pred = model(x)
                loss = l1_loss(pred, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return history


def _make_synth_series(T: int = 32, C: int = 1, H: int = 20, W: int = 20, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = 0.01 * torch.randn(T, C, H, W, generator=g)
    trend = torch.linspace(0, 1, T).view(T, 1, 1, 1)
    out = noise
    out[:, 0] = 0.2 + 0.05 * trend[:, 0] + noise[:, 0]
    return out.float()


if __name__ == "__main__":
    torch.manual_seed(0)
    # basic forward shape check + tiny training smoke test
    series = _make_synth_series(T=24, C=1)
    pairs = make_surface_batches(series, lookback=10, stride=1)
    xb, yb = pairs[0]

    model = ConvLSTM(in_channels=1, hidden_channels=16, kernel_size=3, num_layers=1)
    with torch.no_grad():
        yhat = model(xb)
        assert yhat.shape == yb.shape
    print("✅ forward shape ok:", tuple(yhat.shape))

    batches = [SurfaceBatch(x_seq=x, y_next=y) for x, y in pairs[:6]]
    hist = train_model(model, batches, epochs=3, lr=1e-3)
    print("✅ train smoke loss:", hist)

    # 추가: ConvTF/SAConvLSTM forward 체크
    sa = SAConvLSTM(in_channels=1, hidden_channels=16, kernel_size=3, num_layers=1)
    tf = ConvTF(in_channels=1, d_model=16, heads=4, num_layers=1, sffn_depth=2)
    with torch.no_grad():
        assert sa(xb).shape == yb.shape
        assert tf(xb).shape == yb.shape
    print("✅ SAConvLSTM/ConvTF forward ok")

    # 추가: PIConvTF compute_loss backward smoke
    pi = PIConvTF(d_model=16, heads=4, num_layers=1, lambda_phys=0.01, sffn_depth=2)
    x_pi = torch.randn(1, 10, 5, 20, 20) * 0.01
    y_pi = torch.randn(1, 1, 20, 20) * 0.01 + 0.2
    market_next = {
        "tau": torch.rand(1, 1, 20, 20),
        "S": torch.rand(1, 1, 20, 20) * 100 + 3000,
        "r": torch.rand(1, 1, 20, 20) * 0.05,
        "K": torch.rand(1, 1, 20, 20) * 100 + 3000,
    }
    loss_obj = pi.compute_loss(x_pi, y_pi, market_next=market_next)
    loss_obj.total_loss.backward()
    print("✅ PIConvTF backward ok")

