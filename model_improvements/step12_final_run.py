"""model_improvements/step12_final_run.py
Part 2, step 12: the combined run with all improvements
(learnable D + learnable IC width + adaptive gradient-norm weights + data/PDE/BC/IC/wall terms).
Press Run, no flags. Trains missing jobs in parallel and caches them in model_improvements/results/step12_ic/.
NOTE: this combined run failed the synthetic recovery test (see step12c_gate.py / step12d_real_wells.py for the
settings actually used). Figures are NOT made here: see improvement_figure_generation.py."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import D_NOM, make_synthetic_well, nominal_ic
from step10b_learnable_IC import train_inverse_ic
from step11_adaptive_weights import GradNormWeights
from config import Domain
from data import make_case, load_well, all_wells
from analytical import estimate_sft

# ----------------------------- settings -----------------------------
EPOCHS, N_P, SEED = 8000, 2000, 0
SYN = {"syn_A": dict(D_mult=0.5, ell=1.0), "syn_B": dict(D_mult=1.0, ell=0.5)}   # synthetic truths
SYN_NOISE = 0.1
SHARE_REAL = dict(ic=0.25, wall=0.25)       # my assumption: IC/wall terms get less than data/PDE/BC
OUT = HERE / "results" / "step12_ic"
dom = Domain()
BASE_OVERSHOOT = dict(MXCO1=14.56, MXCO2=7.97, SGIL=3.52, USAM=2.61, PHIL=44.69, JAPN=1.03, **{"CH-A1": 10.73},
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
        spec = SYN[name]
        case = make_case("SGIL")
        well, _ = make_synthetic_well(case, load_well("SGIL", drop_dt0=True), D_NOM * spec["D_mult"],
                                      SYN_NOISE, 0, nominal_ic(case, spec["ell"]))
        wt = GradNormWeights()
    else:
        case, well = well_case(name)
        wt = GradNormWeights(share=SHARE_REAL)
    model, ic, hist, secs = train_inverse_ic(case, well, D0=D_NOM, epochs=EPOCHS, n_p=N_P, seed=SEED,
                                             weighter=wt, ell_init=1.0, verbose=False)
    rg, tg = np.linspace(dom.r_w, dom.r_max, 200), np.linspace(0, case.t_max_h, 60)
    R, Tt = np.meshgrid(rg, tg)
    with torch.no_grad():
        T = model(torch.tensor(R.ravel()).reshape(-1, 1), torch.tensor(Tt.ravel()).reshape(-1, 1)).numpy().reshape(R.shape)
        pred = model(torch.full((len(well.t_h), 1), dom.r_w, dtype=torch.float64),
                     torch.tensor(well.t_h, dtype=torch.float64).reshape(-1, 1)).numpy().ravel()
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt"); torch.save(ic.state_dict(), d / "ic.pt")
    np.savez(d / "history.npz", **hist)
    m = dict(label=name, D_ratio_nom=float(model.D.item()) / D_NOM, ell_mult=float(ic.mult.item()),
             rmse=float(np.sqrt(np.mean((pred - well.bht) ** 2))), overshoot=float(T.max() - case.sft),
             final_loss=float(hist["total"][-1]), minutes=(time.time() - t0) / 60,
             final_weights={k: float(v) for k, v in wt.w.items()})
    if kind == "syn":
        m.update(D_true_ratio=SYN[name]["D_mult"], ell_true=SYN[name]["ell"])
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    return name, m


def train_missing():
    import multiprocessing as mp
    jobs = [("syn", n) for n in SYN] + [("well", n) for n in all_wells()]
    jobs = [j for j in jobs if not (OUT / j[1] / "metrics.json").exists()]
    hdr("TRAINING STAGE")
    print(f"{len(jobs)} jobs to train on {os.cpu_count()} cores (cached jobs are skipped)", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(os.cpu_count() or 2, len(jobs))) as pool:
        for label, m in pool.imap_unordered(_job, jobs):
            print(f"  done {label:6s} D/D_nom = {m['D_ratio_nom']:.2f}, ell_mult = {m['ell_mult']:.2f}, "
                  f"data RMSE {m['rmse']:.2f} C ({m['minutes']:.1f} min)", flush=True)
    print(f"training wall time {(time.time() - t0) / 60:.1f} min", flush=True)


def report():
    hdr("RESULTS")
    ms = {p.parent.name: json.loads(p.read_text()) for p in OUT.glob("*/metrics.json")}
    print("Synthetic recovery (both parameters start at 1.0 x nominal):")
    print(f"{'job':6s} {'D true':>7s} {'D learned':>10s} {'ell true':>9s} {'ell learned':>12s} {'data RMSE':>10s}")
    for n in SYN:
        s = ms[n]
        print(f"{n:6s} {s['D_true_ratio']:7.2f} {s['D_ratio_nom']:10.2f} {s['ell_true']:9.2f} {s['ell_mult']:12.2f} {s['rmse']:10.3f}")
    print("   Compare with the step 10b profile: if its contrast was below the noise, expect a wrong (D, ell) pair that fits equally well.")
    print(f"\n{'well':6s} {'D/D_nom':>8s} {'ell_mult':>9s} {'RMSE':>6s} {'overshoot':>10s} {'baseline':>9s} {'w_p':>8s} {'w_wall':>8s}")
    for n in all_wells():
        m = ms[n]
        print(f"{n:6s} {m['D_ratio_nom']:8.2f} {m['ell_mult']:9.2f} {m['rmse']:6.2f} {m['overshoot']:+10.2f} "
              f"{BASE_OVERSHOOT[n]:+9.2f} {m['final_weights']['p']:8.3g} {m['final_weights']['wall']:8.3g}")
    print("Real-well D and ell are effective, jointly confounded values (wall and IC model error is absorbed), not measurements.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_missing()
    report()
    print("\nstep 12 finished")


if __name__ == "__main__":      # required: multiprocessing (spawn) on macOS
    main()

