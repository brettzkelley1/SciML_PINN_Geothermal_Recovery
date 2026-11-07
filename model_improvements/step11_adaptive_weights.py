"""model_improvements/step11_adaptive_weights.py
Part 2, step 11: adaptive loss weighting by gradient-norm balancing.
Trains NOTHING beyond a 60-epoch wiring check. Run from PyCharm (no flags).
Use in the final run:  train_inverse(case, well, D0, epochs, weighter=GradNormWeights())
Figures are NOT made here: see improvement_figure_generation.py."""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import (KEYS, D_NOM, WELL, FixedWeights, build_inverse_model,
                                make_synthetic_well, nominal_ic, train_inverse)
from data import make_case, load_well
from losses import LossWeights, SamplingConfig, Sampler, compute_losses

hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)
ALL = LossWeights(m=1, p=1, bc=1, ic=1, wall=1)      # nonzero so every term is computed


class GradNormWeights:
    """Weighter hook: weighter(step, parts, model) -> dict of weights.
    Every `update_every` epochs: w_k_hat = share_k * mean_j(||g_j||) / ||g_k||, g_k = grad of term k wrt the
    network weights (log_D excluded). Then clip, and smooth in log space. Gradients are not propagated into the
    weights themselves (plain floats), so D's learning is not distorted."""

    def __init__(self, init=None, keys=KEYS, update_every=100, alpha=0.9,
                 clip=(1e-3, 1e3), share=None, exclude=("log_D",)):
        self.keys = tuple(keys)
        self.w = dict(FixedWeights().w if init is None else init)
        self.every, self.alpha, self.clip = int(update_every), float(alpha), clip
        self.share = {k: 1.0 for k in self.keys}
        if share:
            self.share.update(share)
        self.exclude = set(exclude)
        self.started = False
        self.history = []                      # (step, weights copy, gradient norms)

    def grad_norms(self, parts, model):
        params = [p for n, p in model.named_parameters() if n not in self.exclude and p.requires_grad]
        out = {}
        for k in self.keys:
            t = parts[k]
            if not t.requires_grad:
                out[k] = 0.0
                continue
            gs = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            s = 0.0
            for g in gs:
                if g is not None:
                    s += float((g ** 2).sum())
            out[k] = s ** 0.5
        return out

    def _update(self, g):
        ok = [k for k in self.keys if np.isfinite(g[k]) and g[k] > 0]
        if len(ok) < 2:
            return
        gbar = float(np.mean([g[k] for k in ok]))
        lo, hi = self.clip
        for k in ok:
            target = float(np.clip(self.share[k] * gbar / g[k], lo, hi))
            if self.started:
                target = float(np.exp(self.alpha * np.log(self.w[k]) + (1 - self.alpha) * np.log(target)))
            self.w[k] = target
        self.started = True

    def __call__(self, step, parts, model):
        if step % self.every == 0:
            g = self.grad_norms(parts, model)
            self._update(g)
            self.history.append((step, dict(self.w), g))
        return self.w


# ----------------------------- 1. exact toy tests of the weighter -----------------------------
def toy(scale_a, scale_b):
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 1, dtype=torch.float64)
    x1, x2 = torch.randn(8, 3, dtype=torch.float64), torch.randn(8, 3, dtype=torch.float64)
    return m, {"a": (scale_a * m(x1)).pow(2).mean(), "b": (scale_b * m(x2)).pow(2).mean()}


def toy_tests():
    hdr("1. Toy tests of the weighter (exact)")
    mk = lambda **kw: GradNormWeights(init={"a": 1.0, "b": 1.0}, keys=("a", "b"), **kw)
    # equal shares: weighted gradient norms must be equal
    m, parts = toy(10.0, 0.1); wt = mk(); wt(0, parts, m)
    gw = wt.grad_norms({k: wt.w[k] * parts[k] for k in parts}, m)
    print(f"equal shares : weights {wt.w['a']:.3g}/{wt.w['b']:.3g}, weighted grad norms {gw['a']:.4g} vs {gw['b']:.4g}")
    assert abs(gw["a"] - gw["b"]) < 1e-9 * gw["a"]
    # share a=2, b=1: weighted gradient norm of a must be twice that of b
    m, parts = toy(10.0, 0.1); wt = mk(share={"a": 2.0}); wt(0, parts, m)
    gw = wt.grad_norms({k: wt.w[k] * parts[k] for k in parts}, m)
    print(f"share a=2    : weighted grad norm ratio a/b = {gw['a'] / gw['b']:.6f} (expect 2)")
    assert abs(gw["a"] / gw["b"] - 2.0) < 1e-9
    # clipping: a 1e10 gradient imbalance must not give weights outside the clip range
    m, parts = toy(1e5, 1.0); wt = mk(); wt(0, parts, m)
    print(f"clip         : weights {wt.w['a']:.3g}, {wt.w['b']:.3g} (range {wt.clip})")
    assert all(wt.clip[0] <= v <= wt.clip[1] for v in wt.w.values())
    # smoothing: second update moves only part of the way (geometric moving average)
    m, parts = toy(10.0, 0.1); wt = mk(update_every=1); wt(0, parts, m); w0 = dict(wt.w)
    m2, parts2 = toy(1.0, 1.0); wt(1, parts2, m2)
    print(f"smoothing    : w_a {w0['a']:.3g} -> {wt.w['a']:.3g} (full jump would be ~1)")
    assert min(w0["a"], 1.0) < wt.w["a"] < max(w0["a"], 1.0)
    print("OK")


# ----------------------------- 2. real PINN at initialisation -----------------------------
def real_model_check():
    hdr("2. Untrained PINN on synthetic SGIL: gradient shares, hand-tuned vs adaptive")
    case, template = make_case(WELL), load_well(WELL, drop_dt0=True)
    syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case))
    model = build_inverse_model(case, D_NOM)
    sc = SamplingConfig(n_p=1000, n_bc=400, n_ic=1000, n_wall=400, r_spacing="log", seed=0)
    batch = Sampler(case, syn, "h", sc).get()
    _, parts = compute_losses(model, batch, ALL)
    wt = GradNormWeights(); wt(0, parts, model)
    g, hand = wt.history[-1][2], FixedWeights().w
    sh_h = {k: hand[k] * g[k] for k in KEYS}; sh_a = {k: wt.w[k] * g[k] for k in KEYS}
    sh_h = {k: v / sum(sh_h.values()) for k, v in sh_h.items()}
    sh_a = {k: v / sum(sh_a.values()) for k, v in sh_a.items()}
    print(f"{'term':5s} {'loss':>10s} {'|grad|':>10s} {'hand w':>8s} {'hand share':>11s} {'adapt w':>10s} {'adapt share':>12s}")
    for k in KEYS:
        print(f"{k:5s} {parts[k].item():10.2e} {g[k]:10.2e} {hand[k]:8.0e} {sh_h[k]:11.3f} {wt.w[k]:10.3g} {sh_a[k]:12.3f}")
    print("share = fraction of the total weighted gradient norm. Adaptive is equal by construction;")
    print("the 'hand share' column shows what the step 9 weights (w_p = 1e-3) implied at initialisation.")
    assert all(np.isfinite(list(g.values())))


# ----------------------------- 3. wiring check and overhead -----------------------------
def wiring_and_overhead():
    hdr("3. Wiring check (60 epochs, NOT a real training run) and update overhead")
    case, template = make_case(WELL), load_well(WELL, drop_dt0=True)
    syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case))
    model = build_inverse_model(case, 3 * D_NOM)
    sc = SamplingConfig(n_p=1000, n_bc=400, n_ic=1000, n_wall=400, r_spacing="log", seed=0)
    batch = Sampler(case, syn, "h", sc).get()
    wt, t_ep, t_gn = GradNormWeights(), [], []
    for _ in range(3):
        model.zero_grad(); t0 = time.perf_counter()
        _, parts = compute_losses(model, batch, ALL)
        sum(parts[k] for k in KEYS).backward(); t_ep.append(time.perf_counter() - t0)
    for _ in range(3):
        _, parts = compute_losses(model, batch, ALL); t0 = time.perf_counter()
        wt.grad_norms(parts, model); t_gn.append(time.perf_counter() - t0)
    ep, gn = float(np.median(t_ep)), float(np.median(t_gn))
    print(f"one epoch {ep * 1e3:.0f} ms; one weight update {gn * 1e3:.0f} ms -> overhead at update_every=100: "
          f"{100 * gn / (100 * ep):.1f}% (at n_p=1000)")

    wt = GradNormWeights(update_every=20)
    _, hist, secs = train_inverse(case, syn, D0=3 * D_NOM, epochs=60, n_p=500, weighter=wt, verbose=False)
    print(f"60 epochs in {secs:.1f} s; D {hist['D'][0]:.3e} -> {hist['D'][-1]:.3e}; "
          f"loss {hist['total'][0]:.2e} -> {hist['total'][-1]:.2e}")
    print(f"{'step':>5s} " + " ".join(f"{k:>9s}" for k in KEYS) + "   spread of w*|grad| (max/min)")
    for step, w, g in wt.history:
        wg = [w[k] * g[k] for k in KEYS]
        print(f"{step:5d} " + " ".join(f"{w[k]:9.3g}" for k in KEYS) + f"   {max(wg) / min(wg):.2f}")
    assert np.isfinite(hist["total"]).all() and len(wt.history) == 3
    assert any(abs(wt.history[-1][1][k] - FixedWeights().w[k]) > 1e-12 for k in KEYS), "weights never changed"
    print("OK: weighter runs inside train_inverse and weights adapt")


def main():
    toy_tests()
    real_model_check()
    wiring_and_overhead()
    print("\nstep 11 checks finished (no full training was run)")


if __name__ == "__main__":
    main()

