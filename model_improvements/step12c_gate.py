"""model_improvements/step12c_gate.py
Gate calibration: 5 settings x {syn_A (D=0.5x), syn_B (D=2x)} in parallel. IC width fixed at truth.
Trains no real wells. Results cache in results/step12c/. Press Run, no flags."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import D_NOM, make_synthetic_well, nominal_ic
from step10b_learnable_IC import train_inverse_ic
from step11_adaptive_weights import GradNormWeights
from data import make_case, load_well

N_P, SEED, NOISE, TOL = 2000, 0, 0.1, 0.15
TRUTH = {"A": 0.5, "B": 2.0}
CLIP = (1e-3, 1e1)
# name: (epochs, lr_D, share overrides)
VARIANTS = {
    "v1_16k_lrD2e-3":      (16000, 2e-3, {}),
    "v2_16k_lrD1e-2":      (16000, 1e-2, {}),
    "v3_8k_lrD1e-2":       (8000,  1e-2, {}),
    "v4_16k_lrD2e-3_ic3":  (16000, 2e-3, {"ic": 3.0}),
    "v5_16k_lrD1e-2_ic3":  (16000, 1e-2, {"ic": 3.0}),
}
OUT = HERE / "results" / "step12c"
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


def _job(job):
    var, tr = job
    torch.set_num_threads(1)
    epochs, lr_D, share = VARIANTS[var]
    t0 = time.time()
    case = make_case("SGIL")
    well, _ = make_synthetic_well(case, load_well("SGIL", drop_dt0=True), D_NOM * TRUTH[tr], NOISE, 0, nominal_ic(case, 1.0))
    wt = GradNormWeights(clip=CLIP, share=share or None)
    model, ic, hist, _ = train_inverse_ic(case, well, D0=D_NOM, epochs=epochs, n_p=N_P, seed=SEED, weighter=wt,
                                          lr_D=lr_D, ell_init=1.0, learn_ell=False, verbose=False)
    with torch.no_grad():
        pred = model(torch.full((len(well.t_h), 1), case.domain.r_w, dtype=torch.float64),
                     torch.tensor(well.t_h, dtype=torch.float64).reshape(-1, 1)).numpy().ravel()
    D = hist["D"] / D_NOM
    d = OUT / f"{var}_{tr}"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "history.npz", **hist)
    m = dict(var=var, truth=tr, D_ratio=float(D[-1]), rmse=float(np.sqrt(np.mean((pred - well.bht) ** 2))),
             drift_last1000=float(D[-1] - D[-1000]), minutes=(time.time() - t0) / 60)
    (d / "metrics.json").write_text(json.dumps(m))
    return m


def main():
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, t) for v in VARIANTS for t in TRUTH if not (OUT / f"{v}_{t}" / "metrics.json").exists()]
    hdr("GATE CALIBRATION")
    print(f"{len(jobs)} jobs to train on {os.cpu_count()} cores (cached skipped)", flush=True)
    if jobs:
        t0 = time.time()
        with mp.get_context("spawn").Pool(min(os.cpu_count() or 2, len(jobs))) as pool:
            for m in pool.imap_unordered(_job, jobs):
                print(f"  done {m['var']} {m['truth']} ({m['minutes']:.1f} min)", flush=True)
        print(f"wall time {(time.time() - t0) / 60:.1f} min", flush=True)
    hdr("RESULTS (pass = both D_hat/D_true within 15% and data RMSE < 0.3 C)")
    print(f"{'variant':22s} | {'A: D_hat/D_true':>15s} {'RMSE':>6s} {'drift/1k':>9s} | {'B: D_hat/D_true':>15s} {'RMSE':>6s} {'drift/1k':>9s} | verdict")
    for v in VARIANTS:
        ms = {t: json.loads((OUT / f"{v}_{t}" / "metrics.json").read_text()) for t in TRUTH}
        rec = {t: ms[t]["D_ratio"] / TRUTH[t] for t in TRUTH}
        ok = all(abs(rec[t] - 1) < TOL and ms[t]["rmse"] < 0.3 for t in TRUTH)
        print(f"{v:22s} | {rec['A']:15.3f} {ms['A']['rmse']:6.3f} {ms['A']['drift_last1000']:+9.3f} | "
              f"{rec['B']:15.3f} {ms['B']['rmse']:6.3f} {ms['B']['drift_last1000']:+9.3f} | {'PASS' if ok else 'FAIL'}")
    print("drift/1k = change in D/D_nom over the last 1000 epochs; near 0 means converged, large means still moving.")


if __name__ == "__main__":
    main()

