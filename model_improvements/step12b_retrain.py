"""model_improvements/step12b_retrain.py
Retrain with: learnable D only (IC width fixed), weights capped at 10, D lr 2e-3, D clamped to [1/20, 20] x nominal.
Gate: the two synthetic jobs run first; the 14 wells only run if D is recovered within 15%.
Results cache in results/step12b/. Press Run, no flags."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import D_NOM, make_synthetic_well, nominal_ic
from step10b_learnable_IC import train_inverse_ic
from step11_adaptive_weights import GradNormWeights
from config import Domain
from data import make_case, load_well, all_wells
from analytical import estimate_sft

EPOCHS, N_P, SEED = 8000, 2000, 0
SYN = {"syn_A": 0.5, "syn_B": 2.0}          # true D / nominal; IC width is correct (1.0) and fixed
SYN_NOISE, WEIGHT_CLIP = 0.1, (1e-3, 1e1)
REAL_ELL = 0.5                              # fixed IC width on real wells: my judgement (steps 4, 10 favour narrow)
SHARE_REAL = dict(ic=0.25, wall=0.25)
GATE_TOL = 0.15
OUT = HERE / "results" / "step12b"
dom = Domain()
BASE = dict(MXCO1=14.56, MXCO2=7.97, SGIL=3.52, USAM=2.61, PHIL=44.69, JAPN=1.03, **{"CH-A1": 10.73},
            **{"CH-A2": 0.01}, ITAL=0.01, COST=0.06, SHBE=0.55, CLAH=0.0, **{"CH-A4": 0.25}, **{"KJ-21": 36.31})
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


def well_case(name):
    w0 = load_well(name)
    sft = None if w0.group == "train" else estimate_sft(w0, "ascencio")["sft"]
    return make_case(name, sft=sft), load_well(name, drop_dt0=(name == "SGIL"))


def _job(job):
    kind, name = job
    torch.set_num_threads(1)
    t0 = time.time()
    if kind == "syn":
        case = make_case("SGIL")
        well, _ = make_synthetic_well(case, load_well("SGIL", drop_dt0=True), D_NOM * SYN[name], SYN_NOISE, 0, nominal_ic(case, 1.0))
        wt, ell = GradNormWeights(clip=WEIGHT_CLIP), 1.0
    else:
        case, well = well_case(name)
        wt, ell = GradNormWeights(clip=WEIGHT_CLIP, share=SHARE_REAL), REAL_ELL
    model, ic, hist, _ = train_inverse_ic(case, well, D0=D_NOM, epochs=EPOCHS, n_p=N_P, seed=SEED, weighter=wt,
                                          ell_init=ell, learn_ell=False, verbose=False)
    rg, tg = np.linspace(dom.r_w, dom.r_max, 200), np.linspace(0, case.t_max_h, 60)
    R, Tt = np.meshgrid(rg, tg)
    with torch.no_grad():
        T = model(torch.tensor(R.ravel()).reshape(-1, 1), torch.tensor(Tt.ravel()).reshape(-1, 1)).numpy().reshape(R.shape)
        pred = model(torch.full((len(well.t_h), 1), dom.r_w, dtype=torch.float64),
                     torch.tensor(well.t_h, dtype=torch.float64).reshape(-1, 1)).numpy().ravel()
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt"); np.savez(d / "history.npz", **hist)
    m = dict(label=name, D_ratio_nom=float(model.D.item()) / D_NOM, rmse=float(np.sqrt(np.mean((pred - well.bht) ** 2))),
             overshoot=float(T.max() - case.sft), minutes=(time.time() - t0) / 60,
             final_weights={k: float(v) for k, v in wt.w.items()})
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    return name, m


def run(jobs):
    import multiprocessing as mp
    jobs = [j for j in jobs if not (OUT / j[1] / "metrics.json").exists()]
    print(f"{len(jobs)} jobs to train (cached skipped)", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(os.cpu_count() or 2, len(jobs))) as pool:
        for label, m in pool.imap_unordered(_job, jobs):
            print(f"  done {label:6s} D/D_nom = {m['D_ratio_nom']:.2f}, RMSE {m['rmse']:.2f} C ({m['minutes']:.1f} min)", flush=True)
    print(f"wall time {(time.time() - t0) / 60:.1f} min", flush=True)


def load(name):
    return json.loads((OUT / name / "metrics.json").read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hdr("GATE: synthetic recovery of D (IC width fixed at truth)")
    run([("syn", n) for n in SYN])
    ok = True
    for n, truth in SYN.items():
        m = load(n); rec = m["D_ratio_nom"] / truth
        good = abs(rec - 1) < GATE_TOL and m["rmse"] < 0.3
        ok &= good
        print(f"{n}: D true {truth:.2f}x, learned {m['D_ratio_nom']:.2f}x -> D_hat/D_true {rec:.3f}, RMSE {m['rmse']:.3f} C  "
              f"{'PASS' if good else 'FAIL'}")
    if not ok:
        print("\nGate failed: D is not recoverable with these settings. Real wells NOT trained. Paste this output.")
        return
    hdr("REAL WELLS")
    run([("well", n) for n in all_wells()])
    print(f"\n{'well':6s} {'D/D_nom':>8s} {'RMSE':>6s} {'overshoot':>10s} {'baseline':>9s} {'w_p':>7s} {'w_wall':>7s}")
    for n in all_wells():
        m = load(n)
        print(f"{n:6s} {m['D_ratio_nom']:8.2f} {m['rmse']:6.2f} {m['overshoot']:+10.2f} {BASE[n]:+9.2f} "
              f"{m['final_weights']['p']:7.3g} {m['final_weights']['wall']:7.3g}")
    print(f"IC width fixed at {REAL_ELL}; real-well D is an effective value conditional on that choice, not a measurement.")


if __name__ == "__main__":
    main()

