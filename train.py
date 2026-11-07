import time
from dataclasses import replace
import numpy as np
import torch
from config import PINNConfig, RockProps, H_TO_S
from data import make_case, load_well
from pinn import PINN, pde_residual
from losses import LossWeights, SamplingConfig, Sampler, compute_losses

DT = torch.float64


def _mode(s):
    return {True: "full", False: "raw"}.get(s, s)   # bool kept for benchmark.py


def build_model(case, unit, seed, lr, epochs, scaled="raw", dtype=torch.float64):
    mode = _mode(scaled)
    assert mode in ("raw", "out", "full"), mode
    cfg = replace(PINNConfig(), seed=seed, lr=lr, epochs=epochs)
    tm = H_TO_S if unit == "s" else 1.0
    kw = {}
    if mode in ("out", "full"):      # NOT paper mode
        kw.update(out_scale=case.sft - case.t0_bht, out_shift=case.t0_bht)
    if mode == "full":               # NOT paper mode
        kw.update(in_scale=(case.domain.r_max, case.t_max_h * tm))
    return PINN(cfg, time_unit=unit, dtype=dtype, **kw)


def train_one(name, unit, seed=0, epochs=15_000, lr=1e-3, n_p=5000, n_bc=200,
              sft_method="ascencio", scaled=False, log_every=1000, verbose=True,
              dtype=torch.float64, device="cpu", scale_mode=None, drop_dt0=False):
    mode = _mode(scale_mode if scale_mode is not None else scaled)
    case, well = make_case(name, method=sft_method), load_well(name, drop_dt0)
    model = build_model(case, unit, seed, lr, epochs, mode, dtype).to(device)
    batch = Sampler(case, well, unit, SamplingConfig(n_p=n_p, n_bc=n_bc, seed=seed)).get()
    batch = {k: {kk: vv.to(device=device, dtype=dtype) for kk, vv in d.items()}
             for k, d in batch.items()}
    weights = LossWeights()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    H = torch.full((4, epochs), float("nan"), device=device)
    t0, n_done = time.time(), 0
    for ep in range(epochs):
        opt.zero_grad()
        tot, parts = compute_losses(model, batch, weights)
        H[:, ep] = torch.stack([tot.detach(), parts["m"].detach(),
                                parts["p"].detach(), parts["bc"].detach()]).float()
        if ep % 100 == 0 and not torch.isfinite(tot):
            print(f"  !! non-finite loss at epoch {ep}; stopping"); break
        tot.backward(); opt.step(); n_done = ep + 1
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:6d}: total {H[0,ep].item():.3e} (m {H[1,ep].item():.2e}, "
                  f"p {H[2,ep].item():.2e}, bc {H[3,ep].item():.2e})")
    Hn = H.cpu().numpy()
    hist = {k: Hn[i, :n_done] for i, k in enumerate(("total", "m", "p", "bc"))}
    model = model.to("cpu").double()
    return dict(case=case, well=well, model=model, unit=unit, seed=seed, mode=mode,
                hist=hist, seconds=time.time() - t0, n_done=n_done)


def evaluate(res):
    model, case, well, unit = res["model"], res["case"], res["well"], res["unit"]
    tm = H_TO_S if unit == "s" else 1.0
    d, D = case.domain, RockProps().D

    with torch.no_grad():
        r = torch.full((len(well.t_h), 1), d.r_w, dtype=DT)
        t = torch.tensor(well.t_h * tm, dtype=DT).reshape(-1, 1)
        pred = model(r, t).numpy().ravel()
        tb = torch.linspace(0, case.t_max_h * tm, 200, dtype=DT).reshape(-1, 1)
        T_far = model(torch.full_like(tb, d.r_max), tb).numpy().ravel()
    obs = well.bht
    e = pred - obs
    out = dict(
        rmse=float(np.sqrt(np.mean(e ** 2))), max_err=float(np.max(np.abs(e))),
        r2=float(1 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2)),   # paper Eq. 8
        pearson=float(abs(np.corrcoef(pred, obs)[0, 1])),                 # what the paper's "R2" actually is
        far_mae=float(np.mean(np.abs(T_far - case.sft))),
    )

    # collapse detector: range of the wall trace over the window
    with torch.no_grad():
        tw = torch.linspace(0, case.t_max_h * tm, 200, dtype=DT).reshape(-1, 1)
        Tw = model(torch.full_like(tw, d.r_w), tw)
    out["wall_span"] = float(Tw.max() - Tw.min())   # < ~1 C means the net collapsed to a constant

    # PDE residual of the trained net under the CORRECT units
    g = torch.Generator().manual_seed(999)
    rr = d.r_w + (d.r_max - d.r_w) * torch.rand(3000, 1, generator=g, dtype=DT)
    tt = case.t_max_h * tm * torch.rand(3000, 1, generator=g, dtype=DT)
    D_true = D * H_TO_S if unit in ("h", "h_raw") else D
    resid, a, b, c = pde_residual(model, rr, tt, D_true, return_terms=True)
    rms = lambda x: x.detach().pow(2).mean().sqrt().item()
    out["phys_res_rel"] = rms(resid) / max(rms(a), rms(b), rms(c), 1e-30)
    return out





