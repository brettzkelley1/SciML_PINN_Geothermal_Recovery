"""model_improvements/step10_learnable_D.py
Part 2, step 10: learnable thermal diffusivity (identifiability analysis + machinery).
Trains NOTHING by default. The single full training run happens after step 11.
Run from PyCharm (no flags)."""
import sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # import the replication modules
from config import PINNConfig, RockProps, H_TO_S
from data import make_case, load_well, WellData
from fd_solver import solve_fd
from ic import gaussian_ic
from pinn import PINN
from losses import LossWeights, SamplingConfig, Sampler, compute_losses

# ----------------------------- settings -----------------------------
WELL = "SGIL"
RUN_SMOKE = True            # 30-epoch wiring check only (seconds)
OUT = HERE / "results"
D_NOM = RockProps().D
KEYS = ("m", "p", "bc", "ic", "wall")
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


# ----------------------------- FD helpers (no NN) -----------------------------
def wall_trace(case, D, times_h, ic_fn, n_r=200, n_t=1000):
    """FD wall temperature T(r_w, t) at the requested hours (zero-flux wall, Dirichlet far field)."""
    sol = solve_fd(case, n_r=n_r, n_t=n_t, D=D, ic_fn=ic_fn, wall_bc="neumann")
    return np.interp(times_h, sol["t_h"], sol["bht"])


def nominal_ic(case, ell_mult=1.0):
    """Gaussian IC with its width FIXED at the nominal D (so it does not move when D is fitted)."""
    return lambda r: gaussian_ic(r, case, ell_mult=ell_mult, D=D_NOM)


def fit_logD(case, t_obs, y_obs, ic_fn, span=20.0, n_r=200, n_t=1000):
    """1-D least squares for D by FD. Returns (D_hat, rmse, hit_bound)."""
    f = lambda lD: float(np.sum((wall_trace(case, np.exp(lD), t_obs, ic_fn, n_r, n_t) - y_obs) ** 2))
    lo, hi = np.log(D_NOM / span), np.log(D_NOM * span)
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x)), float(np.sqrt(res.fun / len(y_obs))), bool(min(res.x - lo, hi - res.x) < 0.02)


def make_synthetic_well(case, template, D_true, noise_sd, seed, ic_fn):
    """Synthetic WellData: FD wall trace with known D at the template's measurement times (dt > 0) plus noise."""
    t = template.t_h[template.t_h > 0]
    clean = wall_trace(case, D_true, t, ic_fn, n_r=400, n_t=2000)
    noisy = clean + np.random.default_rng(seed).normal(0.0, noise_sd, size=clean.shape)
    return WellData(f"{template.name}_syn", "train", template.t_c_h, t, noisy, {}), clean


# ----------------------------- learnable-D model (used in the final run) -----------------------------
def build_inverse_model(case, D0, seed=0, dtype=torch.float64):
    """Scaled-in/out PINN ('full' baseline mode) with log D as a trainable scalar, initialised at D0 [m^2/s]."""
    cfg = replace(PINNConfig(), seed=seed)
    return PINN(cfg, D0=D0, time_unit="h", learn_D=True,
                in_scale=(case.domain.r_max, case.t_max_h),
                out_scale=case.sft - case.t0_bht, out_shift=case.t0_bht, dtype=dtype)


class FixedWeights:
    """Default weighting (step 9 values). Step 11 replaces this with an adaptive weighter
    having the same call signature: weighter(step, parts, model) -> dict of weights."""
    def __init__(self, m=1.0, p=1e-3, bc=1.0, ic=1.0, wall=1.0):
        self.w = dict(m=m, p=p, bc=bc, ic=ic, wall=wall)
    def __call__(self, step, parts, model):
        return self.w


def make_optimizer(model, lr=1e-3, lr_D=1e-2):
    net = [p for n, p in model.named_parameters() if n != "log_D"]
    return torch.optim.Adam([{"params": net, "lr": lr}, {"params": [model.log_D], "lr": lr_D}])


def train_inverse(case, well, D0, epochs, n_p=5000, seed=0, weighter=None, lr=1e-3, lr_D=1e-2,
                  ell_mult=1.0, log_every=500, verbose=True):
    """Train PINN with data + PDE + BC + IC + wall terms and learnable D. Returns model and history."""
    weighter = FixedWeights() if weighter is None else weighter
    model = build_inverse_model(case, D0, seed)
    sc = SamplingConfig(n_p=n_p, n_bc=400, n_ic=1000, n_wall=400, r_spacing="log", seed=seed)
    batch = Sampler(case, well, "h", sc, ell_mult=ell_mult).get()
    opt = make_optimizer(model, lr, lr_D)
    ALL = LossWeights(m=1, p=1, bc=1, ic=1, wall=1)        # nonzero so every term is computed; real weights below
    hist = {k: [] for k in ("total", "D") + KEYS}
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        _, parts = compute_losses(model, batch, ALL)
        w = weighter(ep, parts, model)
        total = sum(w[k] * parts[k] for k in KEYS)
        total.backward(); opt.step()
        hist["total"].append(total.item()); hist["D"].append(model.D.item())
        for k in KEYS:
            hist[k].append(parts[k].item())
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:6d}: total {total.item():.3e}  D = {model.D.item():.3e} m^2/s "
                  f"(m {parts['m'].item():.1e}, p {parts['p'].item():.1e}, bc {parts['bc'].item():.1e}, "
                  f"ic {parts['ic'].item():.1e}, wall {parts['wall'].item():.1e})", flush=True)
    return model, {k: np.array(v) for k, v in hist.items()}, time.time() - t0


# ----------------------------- analyses -----------------------------
def analysis_D_sensitivity(case, t_obs, ic_fn):
    hdr("1. How much does the wall trace change with D? (FD only)")
    ref = wall_trace(case, D_NOM, t_obs, ic_fn, 400, 2000)
    print(f"data times: {t_obs[0]:.0f}-{t_obs[-1]:.0f} h ({len(t_obs)} points)")
    print(f"{'D / D_true':>10s} {'RMSE vs true trace (C)':>24s} {'max |diff| (C)':>15s}")
    for mult in (0.25, 0.5, 0.8, 1.25, 2.0, 4.0):
        d = wall_trace(case, D_NOM * mult, t_obs, ic_fn, 400, 2000) - ref
        print(f"{mult:10.2f} {np.sqrt(np.mean(d ** 2)):24.3f} {np.abs(d).max():15.3f}")
    print("D is identifiable only if measurement noise is well below these RMSE values.")


def analysis_rmax(case, t_obs, ic_fn):
    hdr("2. Does the far-field radius matter? (FD only)")
    t_all = np.linspace(0, case.t_max_h, 41)[1:]
    base = replace(case, domain=replace(case.domain, r_max=6.5))
    ref_obs, ref_all = wall_trace(base, D_NOM, t_obs, ic_fn), wall_trace(base, D_NOM, t_all, ic_fn)
    print(f"{'r_max (m)':>9s} {'max diff at data times (C)':>27s} {'max diff over window (C)':>26s}")
    for rm in (3.0, 4.5, 6.5, 10.0, 15.0):
        c = replace(case, domain=replace(case.domain, r_max=rm))
        print(f"{rm:9.1f} {np.abs(wall_trace(c, D_NOM, t_obs, ic_fn) - ref_obs).max():27.4f} "
              f"{np.abs(wall_trace(c, D_NOM, t_all, ic_fn) - ref_all).max():26.4f}")


def analysis_fd_recovery(case, template, ic_fn):
    hdr("3. FD inversion on synthetic data: best-case recovery of D")
    print(f"{'D_true/D_nom':>12s} {'noise sd (C)':>12s} {'D_hat/D_true mean':>18s} {'sd':>7s} {'worst':>7s}")
    for mult in (0.5, 1.0, 2.0):
        for sd in (0.0, 0.1, 0.5):
            ratios = []
            for seed in range(5 if sd > 0 else 1):
                syn, _ = make_synthetic_well(case, template, D_NOM * mult, sd, seed, ic_fn)
                dh, _, _ = fit_logD(case, syn.t_h, syn.bht, ic_fn)
                ratios.append(dh / (D_NOM * mult))
            r = np.array(ratios)
            print(f"{mult:12.2f} {sd:12.2f} {r.mean():18.3f} {r.std():7.3f} {r[np.argmax(np.abs(r - 1))]:7.3f}")


def analysis_ic_confound(case, template):
    hdr("4. IC-width confound: data made with IC width x, D fitted assuming nominal width")
    print(f"{'true ell_mult':>13s} {'D_hat/D_true':>13s} {'fit RMSE (C)':>13s}")
    for em in (0.5, 1.0, 2.0):
        syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case, em))
        dh, rm, _ = fit_logD(case, syn.t_h, syn.bht, nominal_ic(case, 1.0))
        print(f"{em:13.2f} {dh / D_NOM:13.3f} {rm:13.3f}")
    print("A ratio far from 1 means D and the unknown IC width trade off; report D with that caveat.")


def analysis_real_fit(case):
    hdr("5. Exploratory: FD fit of D to the real SGIL wall data (dt > 0)")
    w = load_well(WELL, drop_dt0=True)
    print(f"{'IC ell_mult':>11s} {'D_hat/D_nom':>12s} {'fit RMSE (C)':>13s}  note")
    for em in (0.25, 0.5, 1.0):
        dh, rm, hit = fit_logD(case, w.t_h, w.bht, nominal_ic(case, em))
        print(f"{em:11.2f} {dh / D_NOM:12.2f} {rm:13.2f}  {'hit search bound' if hit else ''}")
    print("Uses the zero-flux wall, which step 4 showed misfits SGIL, so D_hat absorbs model error.")


def smoke_test(case, template):
    hdr("6. Wiring check: learnable D receives gradients (30 epochs, not a real training run)")
    syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case))
    model, hist, secs = train_inverse(case, syn, D0=3 * D_NOM, epochs=30, n_p=500, verbose=False)
    print(f"D: {hist['D'][0]:.3e} -> {hist['D'][-1]:.3e} m^2/s  (started at 3x nominal), loss "
          f"{hist['total'][0]:.2e} -> {hist['total'][-1]:.2e}, {secs:.1f} s")
    assert np.isfinite(hist["total"]).all() and hist["D"][-1] != hist["D"][0], "D did not update"
    print("OK: D is trainable and the weighter hook works")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    case = make_case(WELL)
    template = load_well(WELL, drop_dt0=True)
    ic_fn = nominal_ic(case)
    analysis_D_sensitivity(case, template.t_h, ic_fn)
    analysis_rmax(case, template.t_h, ic_fn)
    analysis_fd_recovery(case, template, ic_fn)
    analysis_ic_confound(case, template)
    analysis_real_fit(case)
    if RUN_SMOKE:
        smoke_test(case, template)
    print("\nstep 10 analyses finished (no full training was run)")


if __name__ == "__main__":
    main()

