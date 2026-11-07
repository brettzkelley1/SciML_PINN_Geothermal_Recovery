import time
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from config import H_TO_S
from losses import LossWeights, SamplingConfig, Sampler, compute_losses
from train import build_model

DT = torch.float64


def train_bench(case, well, unit="h", seed=0, epochs=15_000, lr=1e-3, n_p=5000, n_bc=400,
                n_ic=1000, n_wall=400, r_spacing="log", w_p=1e-3, scaled=True, log_every=1000):
    model = build_model(case, unit, seed, lr, epochs, scaled)
    sc = SamplingConfig(n_p=n_p, n_bc=n_bc, n_ic=n_ic, n_wall=n_wall,
                        r_spacing=r_spacing, seed=seed)
    batch = Sampler(case, well, unit, sc).get()
    w = LossWeights(m=0.0, p=w_p, bc=1.0, ic=1.0, wall=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keys = ("total", "p", "bc", "ic", "wall")
    hist = {k: np.full(epochs, np.nan) for k in keys}
    t0, n_done = time.time(), 0
    for ep in range(epochs):
        opt.zero_grad()
        tot, parts = compute_losses(model, batch, w)
        if not torch.isfinite(tot):
            print(f"  !! non-finite loss at epoch {ep}; stopping"); break
        hist["total"][ep] = tot.item()
        for k in keys[1:]:
            hist[k][ep] = parts[k].item()
        tot.backward(); opt.step(); n_done = ep + 1
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  ep {ep:6d}: total {tot.item():.3e} (p {parts['p'].item():.2e}, "
                  f"bc {parts['bc'].item():.2e}, ic {parts['ic'].item():.2e}, "
                  f"wall {parts['wall'].item():.2e})")
    return model, {k: v[:n_done] for k, v in hist.items()}, time.time() - t0


def predict(model, unit, r, t_h, chunk=20_000):
    """Evaluate model on the tensor grid r x t_h. Returns array [len(t_h), len(r)] in deg C."""
    tm = H_TO_S if unit == "s" else 1.0
    R, Tm = np.meshgrid(r, t_h)
    rr = torch.tensor(R.ravel(), dtype=DT).reshape(-1, 1)
    tt = torch.tensor(Tm.ravel() * tm, dtype=DT).reshape(-1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(rr), chunk):
            out.append(model(rr[i:i + chunk], tt[i:i + chunk]))
    return torch.cat(out).numpy().reshape(R.shape)


class FDField:
    """FD solution wrapped as a 'model' (t in hours) to test the comparison plumbing."""
    def __init__(self, sol):
        self.f = RegularGridInterpolator((sol["t_h"], sol["r"]), sol["T"])
    def __call__(self, r, t):
        pts = np.column_stack([t.numpy().ravel(), r.numpy().ravel()])
        return torch.tensor(self.f(pts), dtype=DT).reshape(-1, 1)


def error_metrics(T_pred, sol, case):
    r, t_h, T_fd = sol["r"], sol["t_h"], sol["T"]
    dT = case.sft - case.t0_bht
    e = T_pred - T_fd
    w = r * np.gradient(r); w = w / w.sum()                 # area weights (annulus), not grid weights
    i, j = np.unravel_index(np.argmax(np.abs(e)), e.shape)
    near, early = r < 1.0, t_h <= 0.05 * t_h[-1]
    return dict(
        linf_C=float(np.abs(e).max()), linf_pct_dT=float(100 * np.abs(e).max() / dT),
        linf_at_r_m=float(r[j]), linf_at_t_h=float(t_h[i]),
        rmse_grid_C=float(np.sqrt(np.mean(e ** 2))),
        rmse_area_C=float(np.sqrt(np.mean((e ** 2) @ w))),
        rel_l2=float(np.linalg.norm(e) / np.linalg.norm(T_fd - case.sft)),   # of the deviation from SFT
        ic_err_max_C=float(np.abs(e[0]).max()),
        wall_rmse_C=float(np.sqrt(np.mean(e[:, 0] ** 2))), wall_max_C=float(np.abs(e[:, 0]).max()),
        far_max_C=float(np.abs(e[:, -1]).max()),
        near_wall_rmse_C=float(np.sqrt(np.mean(e[:, near] ** 2))),
        early_rmse_C=float(np.sqrt(np.mean(e[early] ** 2))),
    )

