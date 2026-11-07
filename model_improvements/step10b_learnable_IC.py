"""model_improvements/step10b_learnable_IC.py
Part 2, step 10b: learnable initial-condition width (alongside learnable D).
Trains NOTHING beyond a 60-epoch wiring check. Run from PyCharm (no flags)."""
import sys, time, math
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import (KEYS, D_NOM, WELL, FixedWeights, build_inverse_model,
                                make_synthetic_well, nominal_ic, fit_logD)
from config import H_TO_S
from data import make_case, load_well
from ic import gaussian_ic
from losses import LossWeights, SamplingConfig, Sampler, compute_losses

ELL_MIN, ELL_MAX = 0.1, 4.0     # bounds on the width multiplier
PRIOR_W = 0.01                  # weak prior on log(mult), my assumption
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


class LearnableIC(torch.nn.Module):
    """Gaussian IC with learnable width multiplier m (ell = m * sqrt(4 D_nom t_c)). Kept OUTSIDE the PINN so
    the PINN's parameter list (and the adaptive weighter's gradient norms) are unchanged."""
    def __init__(self, case, init_mult=1.0, lo=ELL_MIN, hi=ELL_MAX, dtype=torch.float64):
        super().__init__()
        self.case = case
        lo_l, hi_l = math.log(lo), math.log(hi)
        self.mid, self.half = (lo_l + hi_l) / 2, (hi_l - lo_l) / 2
        x = (math.log(init_mult) - self.mid) / self.half
        self.raw = torch.nn.Parameter(torch.tensor(math.atanh(x), dtype=dtype))
        self.ell0 = math.sqrt(4.0 * D_NOM * case.t_c_h * H_TO_S)

    @property
    def log_mult(self):
        return self.mid + self.half * torch.tanh(self.raw)

    @property
    def mult(self):
        return torch.exp(self.log_mult)

    def target(self, r):
        c = self.case
        ell = self.mult * self.ell0
        return c.sft - (c.sft - c.t0_bht) * torch.exp(-((r - c.domain.r_w) / ell) ** 2)

    def prior(self):
        return PRIOR_W * (self.log_mult / math.log(2.0)) ** 2


def train_inverse_ic(case, well, D0, epochs, n_p=5000, seed=0, weighter=None, lr=1e-3, lr_D=2e-3,
                     lr_ell=1e-2, ell_init=1.0, learn_ell=True, D_bounds=(1 / 20, 20),
                     log_every=500, verbose=True):
    """learn_ell=False fixes the IC width at ell_init. D_bounds clamps D to [lo, hi] x D_NOM."""
    weighter = FixedWeights() if weighter is None else weighter
    model = build_inverse_model(case, D0, seed)
    ic = LearnableIC(case, ell_init)
    sc = SamplingConfig(n_p=n_p, n_bc=400, n_ic=1000, n_wall=400, r_spacing="log", seed=seed)
    batch = Sampler(case, well, "h", sc).get()
    net = [p for n, p in model.named_parameters() if n != "log_D"]
    groups = [{"params": net, "lr": lr}, {"params": [model.log_D], "lr": lr_D}]
    if learn_ell:
        groups.append({"params": [ic.raw], "lr": lr_ell})
    else:
        ic.raw.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    lo, hi = (math.log(D_NOM * D_bounds[0]), math.log(D_NOM * D_bounds[1])) if D_bounds else (None, None)
    W = LossWeights(m=1, p=1, bc=1, ic=0, wall=1)
    r_ic, t_ic = batch["ic"]["r"], batch["ic"]["t"]
    hist = {k: [] for k in ("total", "D", "ell") + KEYS}
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        _, parts = compute_losses(model, batch, W)
        parts["ic"] = ((model(r_ic, t_ic) - ic.target(r_ic)) ** 2).mean()
        w = weighter(ep, parts, model)
        total = sum(w[k] * parts[k] for k in KEYS) + (ic.prior() if learn_ell else 0.0)
        total.backward(); opt.step()
        if D_bounds:
            with torch.no_grad():
                model.log_D.clamp_(lo, hi)
        hist["total"].append(total.item()); hist["D"].append(model.D.item()); hist["ell"].append(ic.mult.item())
        for k in KEYS:
            hist[k].append(parts[k].item())
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:6d}: total {total.item():.3e}  D = {model.D.item():.3e}  ell_mult = {ic.mult.item():.3f}",
                  flush=True)
    return model, ic, {k: np.array(v) for k, v in hist.items()}, time.time() - t0


# ----------------------------- analyses -----------------------------
def analysis_ell_profile(case, template):
    hdr("1. FD profile: can the data separate IC width from D? (D fitted for each assumed width, noise sd 0.1 C)")
    grid = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
    print(f"{'data':>22s} | " + " ".join(f"{g:6.2f}" for g in grid) + " | best  contrast")
    sets = [(f"synthetic ell={em}", make_synthetic_well(case, template, D_NOM, 0.1, 0, nominal_ic(case, em))[0])
            for em in (0.5, 1.0, 2.0)] + [("real SGIL", load_well(WELL, drop_dt0=True))]
    for name, w in sets:
        rm, dr = [], []
        for g in grid:
            dh, r, _ = fit_logD(case, w.t_h, w.bht, nominal_ic(case, g))
            rm.append(r); dr.append(dh / D_NOM)
        rm = np.array(rm); b = int(np.argmin(rm))
        print(f"{name:>22s} | " + " ".join(f"{v:6.2f}" for v in rm) + f" | {grid[b]:4.2f}  {np.sort(rm)[1] - rm[b]:.3f}")
        print(f"{'  D_hat/D_nom':>22s} | " + " ".join(f"{v:6.2f}" for v in dr) + " |")
    print("Rows: fit RMSE (C) per assumed ell_mult; second line: the D that fit chose. 'contrast' = RMSE gap between")
    print("the best and second-best width. Contrast well above the 0.1 C noise = width identifiable; below = not.")


def wiring_check():
    hdr("2. Wiring check (not a real training run)")
    case, template = make_case(WELL), load_well(WELL, drop_dt0=True)
    r = torch.tensor(np.linspace(case.domain.r_w, 2.0, 50), dtype=torch.float64)
    for m in (1.0, 0.5, 2.0):
        err = np.abs(LearnableIC(case, m).target(r).detach().numpy() - gaussian_ic(r.numpy(), case, ell_mult=m)).max()
        print(f"IC target vs gaussian_ic at ell_mult={m}: max diff {err:.1e}")
        assert err < 1e-9
    li = LearnableIC(case, 1.0)
    with torch.no_grad():
        li.raw.fill_(50.0); hi = li.mult.item(); li.raw.fill_(-50.0); lo = li.mult.item()
    print(f"bounds reached: [{lo:.3f}, {hi:.3f}] (set [{ELL_MIN}, {ELL_MAX}])")
    syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case, 0.5))
    model, ic, h, secs = train_inverse_ic(case, syn, D0=D_NOM, epochs=60, n_p=500, ell_init=1.0, verbose=False)
    print(f"60 epochs in {secs:.1f} s: ell_mult {h['ell'][0]:.3f} -> {h['ell'][-1]:.3f} (truth 0.5), "
          f"D/D_nom {h['D'][0] / D_NOM:.2f} -> {h['D'][-1] / D_NOM:.2f}, loss {h['total'][0]:.2e} -> {h['total'][-1]:.2e}")
    assert np.isfinite(h["total"]).all() and h["ell"][-1] != h["ell"][0], "ell did not update"
    print("OK: IC width receives gradients; nothing here says it will recover the truth")


def main():
    case, template = make_case(WELL), load_well(WELL, drop_dt0=True)
    analysis_ell_profile(case, template)
    wiring_check()
    print("\nstep 10b finished (no full training was run)")


if __name__ == "__main__":
    main()

