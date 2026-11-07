"""model_improvements/step13_validation.py: held-out wells and r_max sensitivity. No training.
Printed tables only; the table images are made by improvement_figure_generation.py."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from improvement_figure_generation import load_improved, field
from run_checks_replicate_NN import load12 as load_baseline
from step12d_real_wells import well_case
from data import all_wells, load_well
from fd_solver import solve_fd
from config import Domain, RockProps, H_TO_S

dom, D = Domain(), RockProps().D
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


def part_a():
    hdr("A. Validation wells: model wall temperature at the end of the window vs known true formation temperature")
    print(f"{'well':6s} {'TFT':>6s} {'SFT used':>9s} {'SFT err%':>9s} {'last BHT':>9s} {'t_max h':>8s} "
          f"{'base T_end':>11s} {'err%':>6s} {'impr T_end':>11s} {'err%':>6s}")
    for n, w in all_wells("val").items():
        base = load_baseline(n); case, _, mi = load_improved(n)
        tb = field(base["model"], [dom.r_w], [case.t_max_h])[0, 0]
        ti = field(mi, [dom.r_w], [case.t_max_h])[0, 0]
        e = lambda v: 100 * (v - w.tft) / w.tft
        print(f"{n:6s} {w.tft:6.1f} {case.sft:9.2f} {e(case.sft):+9.1f} {w.bht[-1]:9.1f} {case.t_max_h:8.0f} "
              f"{tb:11.2f} {e(tb):+6.1f} {ti:11.2f} {e(ti):+6.1f}")
    print("The far-field BC is the estimated SFT, so model T_end is bounded by it: these errors mostly repeat the step 3 SFT error.")


def part_b():
    hdr("B. Far-field radius sensitivity by FD (max change in wall trace vs r_max = 6.5 m, deg C)")
    print(f"{'well':6s} {'t_max h':>8s} {'sqrt(Dt) m':>11s} {'r_max=3':>9s} {'r_max=10':>9s} {'r_max=15':>9s}")
    for n in all_wells():
        case, _ = well_case(n)
        t = np.linspace(0, case.t_max_h, 41)[1:]

        def trace(rm):
            c = replace(case, domain=replace(case.domain, r_max=rm))
            s = solve_fd(c, n_r=200, n_t=1000)
            return np.interp(t, s["t_h"], s["bht"])
        ref = trace(6.5)
        d = [np.abs(trace(rm) - ref).max() for rm in (3.0, 10.0, 15.0)]
        print(f"{n:6s} {case.t_max_h:8.0f} {np.sqrt(D * case.t_max_h * H_TO_S):11.2f} {d[0]:9.3f} {d[1]:9.3f} {d[2]:9.3f}")
    print("Uses the default Gaussian IC and zero-flux wall. A value above ~0.5 C means the 6.5 m boundary matters for that well.")


if __name__ == "__main__":
    part_a(); part_b()
    print("\nstep 13 finished (no training)")



