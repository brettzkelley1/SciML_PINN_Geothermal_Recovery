"""model_improvements/step12d_real_wells.py
Final real-well run with the gate-validated settings (step12c v3): learnable D, IC width fixed at 0.5,
adaptive weights capped at 10, 8000 epochs, lr_D 1e-2, IC/wall shares 1.0. Press Run, no flags.
Results cache in results/step12d/."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import D_NOM
from step10b_learnable_IC import train_inverse_ic
from step11_adaptive_weights import GradNormWeights
from config import Domain
from data import make_case, load_well, all_wells
from analytical import estimate_sft

EPOCHS, N_P, SEED, LR_D = 8000, 2000, 0, 1e-2
CLIP, REAL_ELL = (1e-3, 1e1), 0.5
OUT = HERE / "results" / "step12d"
dom = Domain()
BASE = dict(MXCO1=14.56, MXCO2=7.97, SGIL=3.52, USAM=2.61, PHIL=44.69, JAPN=1.03, **{"CH-A1": 10.73},
            **{"CH-A2": 0.01}, ITAL=0.01, COST=0.06, SHBE=0.55, CLAH=0.0, **{"CH-A4": 0.25}, **{"KJ-21": 36.31})
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


def well_case(name):
    w0 = load_well(name)
    sft = None if w0.group == "train" else estimate_sft(w0, "ascencio")["sft"]
    return make_case(name, sft=sft), load_well(name, drop_dt0=(name == "SGIL"))


def _job(name):
    torch.set_num_threads(1)
    t0 = time.time()
    case, well = well_case(name)
    wt = GradNormWeights(clip=CLIP)
    model, ic, hist, _ = train_inverse_ic(case, well, D0=D_NOM, epochs=EPOCHS, n_p=N_P, seed=SEED, weighter=wt,
                                          lr_D=LR_D, ell_init=REAL_ELL, learn_ell=False, verbose=False)
    rg, tg = np.linspace(dom.r_w, dom.r_max, 200), np.linspace(0, case.t_max_h, 60)
    R, Tt = np.meshgrid(rg, tg)
    with torch.no_grad():
        T = model(torch.tensor(R.ravel()).reshape(-1, 1), torch.tensor(Tt.ravel()).reshape(-1, 1)).numpy().reshape(R.shape)
        pred = model(torch.full((len(well.t_h), 1), dom.r_w, dtype=torch.float64),
                     torch.tensor(well.t_h, dtype=torch.float64).reshape(-1, 1)).numpy().ravel()
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt"); np.savez(d / "history.npz", **hist)
    D = hist["D"] / D_NOM
    m = dict(label=name, D_ratio_nom=float(D[-1]), drift_last1000=float(D[-1] - D[-1000]),
             rmse=float(np.sqrt(np.mean((pred - well.bht) ** 2))), overshoot=float(T.max() - case.sft),
             minutes=(time.time() - t0) / 60, final_weights={k: float(v) for k, v in wt.w.items()})
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    return name, m


def main():
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    names = [n for n in all_wells() if not (OUT / n / "metrics.json").exists()]
    hdr("TRAINING STAGE")
    print(f"{len(names)} jobs to train on {os.cpu_count()} cores (cached skipped)", flush=True)
    if names:
        t0 = time.time()
        with mp.get_context("spawn").Pool(min(os.cpu_count() or 2, len(names))) as pool:
            for n, m in pool.imap_unordered(_job, names):
                print(f"  done {n:6s} D/D_nom = {m['D_ratio_nom']:.2f}, RMSE {m['rmse']:.2f} C ({m['minutes']:.1f} min)", flush=True)
        print(f"wall time {(time.time() - t0) / 60:.1f} min", flush=True)
    hdr("RESULTS")
    print(f"{'well':6s} {'D/D_nom':>8s} {'drift/1k':>9s} {'RMSE':>6s} {'overshoot':>10s} {'baseline':>9s} {'w_p':>7s} {'w_wall':>7s}")
    for n in all_wells():
        m = json.loads((OUT / n / "metrics.json").read_text())
        print(f"{n:6s} {m['D_ratio_nom']:8.2f} {m['drift_last1000']:+9.3f} {m['rmse']:6.2f} {m['overshoot']:+10.2f} "
              f"{BASE[n]:+9.2f} {m['final_weights']['p']:7.3g} {m['final_weights']['wall']:7.3g}")
    print(f"IC width fixed at {REAL_ELL}. Real-well D is an effective value conditional on that choice, not a measurement.")


if __name__ == "__main__":
    main()

