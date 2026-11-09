"""Training losses: adversarial losses and the R1 penalty, segment-balanced gate weights, the Chamfer hinge, and the
local-structure, trend and shape terms."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def ns_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_real.float()).mean() + F.softplus(d_fake.float()).mean()


def ns_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_fake.float()).mean()


def r1_penalty(d_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    grad, = torch.autograd.grad(outputs=d_out.float().sum(), inputs=x_in, create_graph=True, only_inputs=True)
    return grad.float().pow(2).reshape(grad.size(0), -1).sum(1).mean()


def segment_weights(grid_ms: np.ndarray, late_mult: float = 1.0, t_known_ms: float = 0.0) -> np.ndarray:
    """Segment-balanced per-gate weights on normalised log time: early 0.40, mid 0.30, late 0.30*late_mult (each
    segment's weight is spread evenly over its gates; total normalised to 1).  With a late-time extension (t_known_ms
    > 0) the segments are cut on the measured window and the extension joins the late segment."""
    x = np.log10(np.asarray(grid_ms, dtype=np.float64))
    xe = math.log10(float(t_known_ms)) if float(t_known_ms) > 0 else x[-1]
    tau = (x - x[0]) / max(xe - x[0], 1e-12)
    segs = [(tau < 0.15, 0.40), ((tau >= 0.15) & (tau < 0.5), 0.30), (tau >= 0.5, 0.30 * float(late_mult))]
    w = np.zeros_like(tau)
    for m, wt in segs:
        if m.any():
            w[m] = wt / m.sum()
    return (w / w.sum()).astype(np.float32)


def weighted_l1(a: torch.Tensor, b: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    """a, b: (..., T); w_t: (T,) summing to 1 -> mean over leading dims of sum_t w_t |a-b|."""
    return ((a.float() - b.float()).abs() * w_t).sum(dim=-1).mean()


def soft_chamfer(f_fake: torch.Tensor, f_ref: torch.Tensor, radius: float, tau: float = 0.1) -> torch.Tensor:
    """Symmetric Chamfer hinge (precision fake->ref + recall ref->fake, the recall half opposes mode
    collapse).  Per-sample excess distance h = tau*softplus((d - r)/tau) enters as log1p(h): far-off
    samples of an untrained generator cannot dominate the objective, near samples get unit slope."""
    d = torch.cdist(f_fake.float(), f_ref.float())
    h1 = tau * F.softplus((d.min(dim=1).values - radius) / tau)
    h2 = tau * F.softplus((d.min(dim=0).values - radius) / tau)
    return torch.log1p(h1).mean() + torch.log1p(h2).mean()


def local_structure_torch(y: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
    """y: (B, W, T), depth (B, W) -> (B, W-2, T) departure from the neighbour line (spacing aware)."""
    d = depth_m.float()
    w = ((d[:, 1:-1] - d[:, :-2]) / (d[:, 2:] - d[:, :-2]).clamp_min(1e-9)).unsqueeze(-1)
    return y[:, 1:-1].float() - ((1.0 - w) * y[:, :-2].float() + w * y[:, 2:].float())


def local_poly_smoother_torch(depth_m: torch.Tensor, sigma_m: float, deg: int = 2) -> torch.Tensor:
    """(B, W, W) Gaussian local-polynomial smoother rows on each window's stations (depth (B, W), m), the twin of
    local_poly_smoother_np (no gradient: it depends on the station positions only; float64, outside autocast)."""
    with torch.no_grad(), torch.autocast(device_type=depth_m.device.type, enabled=False):
        d = depth_m.double()
        u = (d.unsqueeze(1) - d.unsqueeze(2)) / max(float(sigma_m), 1e-9)   # (B, target, source) in sigmas
        K = torch.exp(-0.5 * u * u)
        q = int(deg) + 1
        V = torch.stack([u ** k for k in range(q)], dim=-1)                  # (B, W, W, q)
        M = torch.einsum("bij,bijp,bijq->bipq", K, V, V) + 1e-12 * torch.eye(q, dtype=u.dtype, device=u.device)
        e1 = torch.zeros(q, 1, dtype=u.dtype, device=u.device)
        e1[0, 0] = 1.0
        c = torch.linalg.solve(M, e1.expand(*M.shape[:2], q, 1))[..., 0]    # (B, W, q) = M_i^-1 e1
        return (K * torch.einsum("bijp,bip->bij", V, c)).float()


def masked_quadratic_residual_torch(y: torch.Tensor, depth_m: torch.Tensor, sigma_m: float, w: torch.Tensor) -> torch.Tensor:
    """y (B, W, T) minus its Gaussian local-quadratic regional fit along the profile (depth (B, W) or (W,), scale
    sigma_m) computed from the cells with w = 1 only (w (B, W, T) in {0, 1}): the labelled anomalies' cells neither
    enter the fit nor bias it, so the residual at the other cells is the anomaly-scale content there.  Differentiable
    in y. Always in float32 (the training loss runs under fp16 autocast; the 3 x 3 solves have no half-precision
    kernel)."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        return _masked_quadratic_residual(y, depth_m, sigma_m, w)


def _masked_quadratic_residual(y: torch.Tensor, depth_m: torch.Tensor, sigma_m: float, w: torch.Tensor) -> torch.Tensor:
    d = depth_m.float()
    if d.dim() == 1:
        d = d.unsqueeze(0)
    u = (d.unsqueeze(1) - d.unsqueeze(2)) / max(float(sigma_m), 1e-9)         # (B|1, target, source) in sigmas
    K = torch.exp(-0.5 * u * u)
    yf, wf = y.float(), w.float()
    Bn = yf.shape[0]
    Ku = [K * u ** k for k in range(5)]
    Sm = [torch.einsum("bij,bjt->bit", Kk.expand(Bn, -1, -1), wf) for Kk in Ku]              # moments (B, W, T)
    Rv = [torch.einsum("bij,bjt->bit", Ku[k].expand(Bn, -1, -1), wf * yf) for k in range(3)]
    A = torch.stack([torch.stack([Sm[0], Sm[1], Sm[2]], -1), torch.stack([Sm[1], Sm[2], Sm[3]], -1),
                     torch.stack([Sm[2], Sm[3], Sm[4]], -1)], -2)                             # (B, W, T, 3, 3)
    eye3 = torch.eye(3, device=y.device, dtype=A.dtype)
    # Such a system (det / (trace / 3)^3 < 1e-6) gets a ridge of 1e-4 x trace / 3 (a ridge fit of a squeezed cell)
    tr3 = (A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]) / 3.0
    a00, a01, a02, a11, a12, a22 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2], A[..., 1, 1], A[..., 1, 2], A[..., 2, 2]
    det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02)
    weak = det < 1e-6 * tr3 ** 3
    ridge = torch.where(weak, 1e-4 * tr3 + 1e-12, 1e-6 * (Sm[0] + 1e-6))
    A = A + ridge.unsqueeze(-1).unsqueeze(-1) * eye3
    # a cell with no unmasked cell within reach (a label target whose whole kernel is label) has an all-but-zero system
    # whose float32 pivots underflow; its residual is never used (the purity map counts off-label cells only), so it is
    # fitted with 0
    empty = Sm[0] < 1e-4
    A = torch.where(empty.unsqueeze(-1).unsqueeze(-1), eye3, A)
    rhs = torch.where(empty.unsqueeze(-1), torch.zeros_like(Sm[0]).unsqueeze(-1), torch.stack(Rv, -1))
    c = torch.linalg.solve(A, rhs.unsqueeze(-1))[..., 0, 0]                                  # fit at the target
    return yf - c


def masked_quadratic_trend_torch(y: torch.Tensor, depth_m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Least-squares quadratic along the profile (per profile / window and time sample) fitted to y (B, W, T) on the
    cells with w = 1 (w (B, W, T) in {0, 1}), evaluated at every station; depth (B, W) or (W,).  A quadratic is
    reproduced exactly by the purity test's local-quadratic fit at any scale and with any mask, so its anomaly-scale
    content is zero.  Differentiable in y; float32 outside autocast."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        d = depth_m.float()
        if d.dim() == 1:
            d = d.unsqueeze(0)
        d = d.expand(y.shape[0], -1)
        mid = 0.5 * (d.amax(dim=1, keepdim=True) + d.amin(dim=1, keepdim=True))
        half = (0.5 * (d.amax(dim=1, keepdim=True) - d.amin(dim=1, keepdim=True))).clamp_min(1e-9)
        u = (d - mid) / half                                                       # (B, W) in [-1, 1]
        V = torch.stack([torch.ones_like(u), u, u * u], dim=-1)                   # (B, W, 3)
        yf, wf = y.float(), w.float()
        A = torch.einsum("bwp,bwt,bwq->btpq", V, wf, V)
        A = A + 1e-6 * (A[..., :1, :1] + 1e-6) * torch.eye(3, device=y.device, dtype=A.dtype)
        r = torch.einsum("bwp,bwt->btp", V, wf * yf)
        c = torch.linalg.solve(A, r.unsqueeze(-1))[..., 0]                         # (B, T, 3)
        return torch.einsum("bwp,btp->bwt", V, c)


def masked_trend_fit_torch(y: torch.Tensor, ref: torch.Tensor, w: torch.Tensor, max_contrast: float = 0.5) -> torch.Tensor:
    """Weighted least-squares fit of y (B, W, T) by a + c (ref - its weighted mean along the window), per window and
    time sample, on the cells weighted w (B, W, T); c is bounded smoothly to +-max_contrast, so 1 + c > 0.  Added to
    a profile whose shape is ref, such a departure changes the level and the depth contrast of ref and keeps its
    direction along the window (background_trend monotone: a "0 anomalies" profile stays a decay along the profile).
    Differentiable in y; float32 outside autocast."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        yf, rf, wf = y.float(), ref.float(), w.float()
        sw = wf.sum(dim=1, keepdim=True).clamp_min(1e-6)
        rc = rf - (wf * rf).sum(dim=1, keepdim=True) / sw
        ym = (wf * yf).sum(dim=1, keepdim=True) / sw
        c = (wf * rc * (yf - ym)).sum(dim=1, keepdim=True) / ((wf * rc * rc).sum(dim=1, keepdim=True) + 1e-9)
        return ym + float(max_contrast) * torch.tanh(c / float(max_contrast)) * rc


def highpass_profile_torch(y: torch.Tensor, depth_m: torch.Tensor, sigma_m: float,
                           S: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Anomaly-scale part of y (B, W, T) along each window's stations: y minus its Gaussian local-quadratic smooth of
    scale sigma_m (the twin of highpass_profile_np).  Smooth backgrounds, linear and quadratic trends give ~0."""
    S = local_poly_smoother_torch(depth_m, sigma_m) if S is None else S
    with torch.autocast(device_type=y.device.type, enabled=False):
        yf = y.float()
        return yf - torch.einsum("bij,bjt->bit", S.float().expand(yf.shape[0], -1, -1), yf)


def structure_projection(ls_dep: torch.Tensor, ls_anchor: torch.Tensor, thr: float,
                         n_bands: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projection coefficient of a departure's local depth structure on the anchor's own, per window and time band:
        beta_b = <LS(dep), LS(A)>_S / <LS(A), LS(A)>_S,   S = {|LS(A)| >= 10 % of the band's peak |LS(A)|}
    ls_dep, ls_anchor: (B, W-2, T) local structures (local_structure_torch).  beta = -0.3 means the departure removes
    30 % of the anchor's local structure in that band (retention 1 + beta = 0.7); a departure orthogonal to it gives 0.
    A band is valid when the anchor's peak |LS| >= thr (same units as the inputs): weaker structure lies inside the
    generator's own variation and its coefficient is ill-conditioned.  Returns (beta, valid), both (B, n_bands)."""
    betas, oks = [], []
    for a, x in zip(torch.tensor_split(ls_anchor.float(), n_bands, dim=2), torch.tensor_split(ls_dep.float(), n_bands, dim=2)):
        amax = a.abs().amax(dim=(1, 2), keepdim=True)
        sup = (a.abs() >= 0.1 * amax).float()
        den = (a * a * sup).sum(dim=(1, 2))
        betas.append((x * a * sup).sum(dim=(1, 2)) / den.clamp_min(1e-20))
        oks.append((amax.view(-1) >= float(thr)) & (den > 0))
    return torch.stack(betas, dim=1), torch.stack(oks, dim=1)


def centred_smooth_l1(b1: torch.Tensor, b2: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Smooth-L1 objective on the CENTRE of a coefficient over latent draws, from two independent draws b1, b2 of the
    same windows.  Value = smooth_l1((b1 + b2) / 2); gradient = the two-sample estimate 0.5 (psi(b2) grad b1 +
    psi(b1) grad b2), psi = clip(b / delta, -1, 1).  With independent draws the expected step is E[psi(b)] grad E[b]:
    it moves the centre (Huber location) of b to 0 and leaves the spread of b over draws free, whereas penalising
    every draw, smooth_l1(b1) + smooth_l1(b2), also shrinks the spread (the latent-driven diversity)."""
    psi1 = (b1.detach() / delta).clamp(-1.0, 1.0)
    psi2 = (b2.detach() / delta).clamp(-1.0, 1.0)
    sur = 0.5 * (psi2 * b1 + psi1 * b2).mean()
    shown = F.smooth_l1_loss(0.5 * (b1 + b2), torch.zeros_like(b1), beta=delta).detach()
    return sur - sur.detach() + shown


# tolerated blur of an intended anomaly's shape by the departure (anomaly_shape_residual): 0.05 = shape correlation
# 0.976 for a departure that does not rescale the anomaly
ANOMALY_SHAPE_TOL = 0.05


def anomaly_shape_residual(dev: torch.Tensor, a: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """Blur of the intended anomaly field a by the departure dev on the anomaly support sup (all (B, W, T)): both
    centred over the support, the part of dev orthogonal to a, as energy relative to a's, r_perp = |dev_perp|^2 /
    |a_c|^2 (B,).  It sets the acceptance check's shape correlation of anomaly + departure with the anomaly, corr =
    (1 + b) / sqrt((1 + b)^2 + r_perp), b = centred slope of dev on a; a departure that is constant or proportional
    to a over the support gives 0.  Windows whose support has no centred variation give 0."""
    n = sup.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    dc = (dev - (dev * sup).sum(dim=(1, 2), keepdim=True) / n) * sup
    ac = (a - (a * sup).sum(dim=(1, 2), keepdim=True) / n) * sup
    aa = (ac * ac).sum(dim=(1, 2))
    ok = aa > 1e-12
    b = (dc * ac).sum(dim=(1, 2)) / aa.clamp_min(1e-12)
    dp = dc - b.view(-1, 1, 1) * ac
    return torch.where(ok, (dp * dp).sum(dim=(1, 2)) / aa.clamp_min(1e-12), torch.zeros_like(aa))


def depth_lowpass_torch(r: torch.Tensor, sigma_st: float) -> torch.Tensor:
    """Gaussian low-pass along the station axis (B, W, T), edge-normalised."""
    B, W, T = r.shape
    rad = int(max(1, math.ceil(3 * sigma_st)))
    k = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=r.device, dtype=torch.float32) / max(sigma_st, 1e-6)) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = r.float().permute(0, 2, 1).reshape(B * T, 1, W)
    num = F.conv1d(F.pad(x, (rad, rad)), k)
    den = F.conv1d(F.pad(torch.ones_like(x), (rad, rad)), k)
    return (num / den.clamp_min(1e-6)).reshape(B, T, W).permute(0, 2, 1)
