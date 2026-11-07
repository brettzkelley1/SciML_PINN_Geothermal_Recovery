"""Replication checks, steps 6-9 and 12 (neural network). No flags: just press Run.
Training stage trains any missing models in parallel (cached), then steps 6-9 and 12 report.
Figures are NOT made here: see replication_figure_generation.py."""
import json, os, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from config import PINNConfig, RockProps, Domain, H_TO_S
from data import make_case, load_well, all_wells
from pinn import PINN, pde_residual
from losses import LossWeights, SamplingConfig, Sampler, compute_losses
from train import build_model, evaluate
from analytical import estimate_sft

# ----------------------------- settings (edit here) -----------------------------
EPOCHS, N_P, SEED = 5000, 2000, 0     # step 12 / step 8 baseline budget (paper: 15000 epochs)
EPOCHS9, N_P9 = 15000, 5000           # step 9 benchmark budget
RUN_STEPS = [6, 7, 8, 9, 12]
HERE = Path(__file__).resolve().parent
OUT12 = HERE / "results" / "step12"
OUT9 = HERE / "results" / "step9_replicate"
DT, D, dom = torch.float64, RockProps().D, Domain()
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


# ----------------------------- shared helpers -----------------------------
def well_case(name):
    """Paper SFT (Ascencio) for training wells; own step-3 Ascencio estimate for validation wells."""
    w0 = load_well(name)
    sft = None if w0.group == "train" else estimate_sft(w0, "ascencio")["sft"]
    return make_case(name, sft=sft), load_well(name, drop_dt0=(name == "SGIL"))


def _worker(job):
    """Top-level so multiprocessing 'spawn' can import it. job = ('well', name) or ('bench', None)."""
    kind, name = job
    torch.set_num_threads(1)
    t0 = time.time()
    if kind == "well":
        case, well = well_case(name)
        model = build_model(case, "h", SEED, 1e-3, EPOCHS, "full")
        batch = Sampler(case, well, "h", SamplingConfig(n_p=N_P, n_bc=200, seed=SEED)).get()
        w, opt = LossWeights(), torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(EPOCHS):
            opt.zero_grad()
            tot, _p = compute_losses(model, batch, w)
            tot.backward(); opt.step()
        d = OUT12 / name; d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "model.pt")
        return f"well {name}", tot.item(), (time.time() - t0) / 60
    # step 9 benchmark: PDE + IC + wall BC, no data, SGIL, same problem as the FD solver
    from fd_solver import solve_fd
    from benchmark import train_bench, predict, error_metrics
    case, well = make_case("SGIL"), load_well("SGIL")
    t1 = time.time(); sol = solve_fd(case, n_r=400, n_t=2000, wall_bc="neumann"); fd_s = time.time() - t1
    model, hist, train_s = train_bench(case, well, "h", SEED, EPOCHS9, n_p=N_P9)
    m = error_metrics(predict(model, "h", sol["r"], sol["t_h"]), sol, case)
    m.update(fd_seconds=fd_s, pinn_train_seconds=train_s, epochs=EPOCHS9, n_p=N_P9, dT_C=case.sft - case.t0_bht)
    OUT9.mkdir(parents=True, exist_ok=True)
    (OUT9 / "metrics.json").write_text(json.dumps(m, indent=2))
    torch.save(model.state_dict(), OUT9 / "model.pt")
    return "step 9 benchmark", float(hist["total"][-1]), (time.time() - t0) / 60


def train_missing():
    import multiprocessing as mp
    jobs = [("well", n) for n in all_wells() if not (OUT12 / n / "model.pt").exists()]
    if not (OUT9 / "metrics.json").exists():
        jobs.append(("bench", None))
    hdr("TRAINING STAGE")
    print(f"{len(jobs)} jobs to train on {os.cpu_count()} cores (cached results are skipped)", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(os.cpu_count() or 2, len(jobs))) as pool:
        for label, loss, mins in pool.imap_unordered(_worker, jobs):
            print(f"  trained {label:18s} final loss {loss:.2e} ({mins:.1f} min)", flush=True)
    print(f"training wall time {(time.time() - t0) / 60:.1f} min", flush=True)


def load12(name):
    case, well = well_case(name)
    model = build_model(case, "h", SEED, 1e-3, EPOCHS, "full")
    model.load_state_dict(torch.load(OUT12 / name / "model.pt")); model = model.double().eval()
    return dict(case=case, well=well, model=model, unit="h")


def field(model, r, t_h):
    R, Tt = np.meshgrid(r, t_h)
    rr, tt = torch.tensor(R.ravel(), dtype=DT).reshape(-1, 1), torch.tensor(Tt.ravel(), dtype=DT).reshape(-1, 1)
    with torch.no_grad():
        return model(rr, tt).numpy().reshape(R.shape)          # [len(t_h), len(r)]


# ----------------------------- step 6 -----------------------------
def step6():
    hdr("STEP 6: architecture and autograd residual")
    g = torch.Generator().manual_seed(123)
    U = lambda lo, hi, n: lo + (hi - lo) * torch.rand(n, 1, generator=g, dtype=DT)
    m = PINN(); n_par = sum(p.numel() for p in m.parameters())
    print(f"params = {n_par} (expect 3021); learn_D adds 1: {sum(p.numel() for p in PINN(learn_D=True).parameters())}")
    assert n_par == 3021
    rel = lambda f, r, t, Dt: (lambda res, a, b, c: res.abs().max().item() / max(a.abs().max(), b.abs().max(), c.abs().max()).item())(
        *pde_residual(f, r, t, Dt, return_terms=True))
    r, t = U(dom.r_w, dom.r_max, 500), U(0, 1.26e6, 500)
    e1 = rel(lambda r, t: r ** 2 + 4 * D * t, r, t, D)
    r, t = U(0.18, 3.0, 500), U(2e4, 1e6, 500)
    e2 = rel(lambda r, t: torch.exp(-r ** 2 / (4 * D * t)) / (4 * np.pi * D * t), r, t, D)
    print(f"residual vs closed forms: {e1:.1e}, {e2:.1e}"); assert e1 < 1e-9 and e2 < 1e-8
    r, t_h = U(dom.r_w, dom.r_max, 100), U(0, 350, 100)
    f_h = lambda r, t: r ** 2 + 4 * D * H_TO_S * t
    bad = pde_residual(f_h, r, t_h, PINN(time_unit="h_raw").D_t).abs().max().item()
    ok = pde_residual(f_h, r, t_h, PINN(time_unit="h").D_t).abs().max().item()
    print(f"unit test: correct {ok:.1e}, D unconverted {bad:.0f} (expect 14396)")
    assert ok < 1e-6 and abs(bad - 14396) < 1e-3
    ml = PINN(learn_D=True, in_scale=(1.0, 3.6e5)); r, t = U(dom.r_w, dom.r_max, 200), U(0, 3.6e5, 200)
    (ml.residual(r, t) ** 2).mean().backward(); print(f"dL/dlogD = {ml.log_D.grad.item():.2e}"); assert ml.log_D.grad.item() != 0


# ----------------------------- step 7 -----------------------------
def step7():
    hdr("STEP 7: sampler and losses")
    case, well = make_case("SGIL"), load_well("SGIL")
    ALL = LossWeights(m=1, p=1, bc=1, ic=1, wall=1)
    b = Sampler(case, well, "s").get()
    assert (b["bc"]["r"] == dom.r_max).all() and (b["bc"]["T"] == case.sft).all() and (b["ic"]["t"] == 0).all()
    assert torch.equal(Sampler(case, well, "h").get()["p"]["t"], Sampler(case, well, "h_raw").get()["p"]["t"])
    model = PINN(time_unit="h"); b = Sampler(case, well, "h", SamplingConfig(n_p=500)).get()
    tot, parts = compute_losses(model, b, ALL)
    with torch.no_grad():
        man = dict(m=((model(b["m"]["r"], b["m"]["t"]).flatten() - torch.tensor(well.bht)) ** 2).mean(),
                   bc=((model(b["bc"]["r"], b["bc"]["t"]) - case.sft) ** 2).mean())
    man["p"] = (pde_residual(model, b["p"]["r"], b["p"]["t"], model.D_t) ** 2).mean()
    for k, v in man.items():
        assert abs(parts[k].item() - v.item()) < 1e-6 * max(abs(v.item()), 1e-30), k
    print("loss terms match independent calculations")
    tot.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("untrained total loss by unit convention (paper Fig. 3 starts ~1.3e9):")
    for name in ("SGIL", "MXCO1"):
        c, wl = make_case(name), load_well(name)
        row = []
        for unit in ("s", "h", "h_raw"):
            bb = Sampler(c, wl, unit).get()
            vals = [compute_losses(PINN(replace(PINNConfig(), seed=s), time_unit=unit), bb, LossWeights())[0].item() for s in range(3)]
            row.append(f"{unit}: {np.median(vals):.1e}")
        print(f"  {name:6s} " + " | ".join(row))


# ----------------------------- step 8 -----------------------------
def step8():
    hdr(f"STEP 8: single-well fits (baseline 'full', {EPOCHS} epochs; SGIL without dt=0 point)")
    print(f"{'well':6s} {'RMSE':>6s} {'|r|':>6s} {'R2':>7s} {'span':>6s} {'physRes':>8s} {'farMAE':>7s}  verdict")
    for name in ("SGIL", "MXCO1"):
        m = evaluate(load12(name))
        ok = m["rmse"] < 2.0 and m["pearson"] > 0.98 and m["wall_span"] > 5.0   # my thresholds, not the paper's
        print(f"{name:6s} {m['rmse']:6.2f} {m['pearson']:6.3f} {m['r2']:7.3f} {m['wall_span']:6.1f} "
              f"{m['phys_res_rel']:8.2e} {m['far_mae']:7.2f}  {'PASS' if ok else 'CHECK'}")
    print("not reproduced: paper Fig. 3 (1.3e9 start); the raw-unit setup collapses (earlier runs)")


# ----------------------------- step 9 -----------------------------
def step9():
    hdr("STEP 9: PINN vs FD benchmark (SGIL, PDE + Gaussian IC + zero-flux wall, no data)")
    from fd_solver import solve_fd
    from benchmark import predict, error_metrics, FDField
    case = make_case("SGIL"); sol = solve_fd(case)
    chk = error_metrics(predict(FDField(sol), "h", sol["r"], sol["t_h"]), sol, case)["linf_C"]
    print(f"plumbing check (FD vs itself): {chk:.1e}"); assert chk < 1e-8
    m = json.loads((OUT9 / "metrics.json").read_text())
    print(f"  {m['epochs']} epochs, n_p={m['n_p']}: Linf {m['linf_C']:.3f} C ({m['linf_pct_dT']:.2f}% of dT) | "
          f"area RMSE {m['rmse_area_C']:.3f} C | IC err {m['ic_err_max_C']:.3f} C | wall RMSE {m['wall_rmse_C']:.3f} C | "
          f"far max {m['far_max_C']:.3f} C | PINN train {m['pinn_train_seconds'] / 60:.1f} min vs FD {m['fd_seconds']:.2f} s")


# ----------------------------- step 12 -----------------------------
def step12():
    hdr(f"STEP 12: all wells, baseline 'full' PINN, {EPOCHS} epochs, n_p={N_P} (figures: replication_figure_generation.py)")
    names = list(all_wells().keys())
    runs = {n: load12(n) for n in names}
    print(f"{'well':6s} {'grp':5s} {'RMSE':>6s} {'|r|':>6s} {'span':>6s} {'overshoot':>9s} {'T(rw,0)':>8s} {'SFT':>7s}")
    rg = np.linspace(dom.r_w, dom.r_max, 200)
    for n, res in runs.items():
        case = res["case"]; m = evaluate(res)
        T = field(res["model"], rg, np.linspace(0, case.t_max_h, 60))
        print(f"{n:6s} {load_well(n).group:5s} {m['rmse']:6.2f} {m['pearson']:6.3f} {m['wall_span']:6.1f} "
              f"{float(T.max() - case.sft):+9.2f} {T[0, 0]:8.1f} {case.sft:7.1f}")
    print("overshoot > 0: net exceeds the far-field SFT, which a diffusion solution cannot do (my check, not the paper's)")


def main():
    train_missing()
    steps = {6: step6, 7: step7, 8: step8, 9: step9, 12: step12}
    for k in RUN_STEPS:
        steps[k]()
    print("\nall NN checks finished")


if __name__ == "__main__":      # required: multiprocessing (spawn) on macOS
    main()


