"""Replication checks, steps 1-5 (no neural network). Usage: python run_checks_replicate.py [--only 1 3 5]"""
import argparse
from dataclasses import replace
import numpy as np
from config import RockProps, Domain, WellCase, H_TO_S
from scaling import Scaler
from data import all_wells, load_well, make_case, PAPER_TABLE1
from analytical import estimate_sft
from fd_solver import solve_fd
from exact_solutions import ExactSolution

ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="+", type=int, default=[1, 2, 3, 4, 5])
args = ap.parse_args()
hdr = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}")


def step1():
    hdr("STEP 1: config and scaling")
    rock, dom = RockProps(), Domain()
    print(f"D = {rock.D:.3e} m^2/s (paper 1.33e-6); r_w = {dom.r_w:.4f} m")
    for t_h in (12, 36, 100, 350):
        L = np.sqrt(rock.D * t_h * H_TO_S)
        print(f"  t={t_h:4d} h: sqrt(Dt)={L:5.2f} m ({100 * L / dom.r_max:4.1f}% of r_max)")
    s = Scaler(WellCase("SGIL", 3.0, 101.73, 62.78, 100.0))
    r, t, T = np.linspace(dom.r_w, dom.r_max, 5), np.array([0., 10., 100.]), np.array([62.78, 90., 101.73])
    assert np.allclose(s.r_from_star(s.r_to_star(r)), r) and np.allclose(s.t_from_star(s.t_to_star(t)), t)
    assert np.allclose(s.theta_to_T(s.T_to_theta(T)), T)
    assert abs(rock.D - 1.333e-6) < 1e-9
    print(f"  r* in [1, {s.r_star_max:.1f}], t*(100 h) = {s.t_star_max:.1f}; round trips OK")


def step2():
    hdr("STEP 2: data module")
    W = all_wells()
    print(f"{'well':6s} {'grp':5s} {'m':>3s} {'tc':>5s} {'dt range':>15s} {'BHT range':>15s}")
    for n, w in W.items():
        print(f"{n:6s} {w.group:5s} {len(w.t_h):3d} {w.t_c_h:5.1f} {w.t_h[0]:6.2f}-{w.t_h[-1]:<7.2f} "
              f"{w.bht.min():6.1f}-{w.bht.max():<7.1f}")
        for i in np.where(np.diff(w.bht) < 0)[0]:
            print(f"    flag: BHT drops {w.bht[i]} -> {w.bht[i + 1]} at dt={w.t_h[i + 1]:.2f} h")
    for n, (asc, hor, *_) in PAPER_TABLE1.items():
        if min(asc, hor) < W[n].bht[-1]:
            print(f"  note: {n} paper SFT below last BHT ({W[n].bht[-1]})")


def step3():
    hdr("STEP 3: Horner / Ascencio vs Canilandi Table 1")
    W = all_wells(); worst, n_ok = 0.0, 0
    print(f"{'well':6s} | {'Asc':>7s} {'paper':>7s} {'|r|':>6s} {'p':>5s} | {'Hor':>7s} {'paper':>7s} {'|r|':>6s} {'p':>5s}")
    for n, (pa, ph, pra, prh) in PAPER_TABLE1.items():
        a, h = estimate_sft(W[n], "ascencio"), estimate_sft(W[n], "horner")
        worst = max(worst, abs(a["sft"] - pa), abs(h["sft"] - ph))
        n_ok += (round(a["r"], 2) == pra) + (round(h["r"], 2) == prh)
        print(f"{n:6s} | {a['sft']:7.2f} {pa:7.2f} {a['r']:6.3f} {pra:5.2f} | {h['sft']:7.2f} {ph:7.2f} {h['r']:6.3f} {prh:5.2f}")
    print(f"max |SFT diff| = {worst:.3f} C; Pearson |r| matches paper 'R2': {n_ok}/20")
    assert worst < 0.01 and n_ok == 20
    print("validation wells (TFT known):")
    for n, w in all_wells("val").items():
        a, h = estimate_sft(w, "ascencio")["sft"], estimate_sft(w, "horner")["sft"]
        print(f"  {n:6s} TFT {w.tft:6.1f} | Asc {a:7.2f} ({100 * (a - w.tft) / w.tft:+5.1f}%) | Hor {h:7.2f} ({100 * (h - w.tft) / w.tft:+5.1f}%)")


def step4():
    hdr("STEP 4: FD solver (SGIL)")
    case, w = make_case("SGIL"), load_well("SGIL")
    sol = solve_fd(case)
    assert sol["T"].max() <= case.sft + 1e-6 and sol["T"].min() >= case.t0_bht - 1e-6
    print("maximum principle OK; FD wall trace vs data:",
          np.round(np.interp(w.t_h[[1, 3, 6, 12]], sol["t_h"], sol["bht"]), 1), "vs", w.bht[[1, 3, 6, 12]])
    ref = solve_fd(case, n_r=1600, n_t=8000); errs = []
    for n_r, n_t in ((200, 1000), (400, 2000), (800, 4000)):
        s = solve_fd(case, n_r=n_r, n_t=n_t)
        errs.append(np.max(np.abs(np.interp(ref["t_h"], s["t_h"], s["bht"]) - ref["bht"])))
        print(f"  n_r={n_r:4d} n_t={n_t:5d}: max |BHT-ref| = {errs[-1]:.2e}")
    assert errs[0] > errs[1] > errs[2]


def step5():
    hdr("STEP 5: FD vs exact eigenmode solutions")
    base = make_case("SGIL")

    def make_test(modes, amps, inner, ref_idx):
        probe = ExactSolution(base, modes, amps, inner)
        case = replace(base, t_max_h=2.0 * probe.tau_s(ref_idx) / H_TO_S)
        return case, ExactSolution(case, modes, amps, inner)

    def run_fd(case, ex, n_r, n_t):
        sol = solve_fd(case, n_r=n_r, n_t=n_t, ic_fn=lambda r: ex(r, 0.0), wall_bc=ex.inner,
                       bht_fn=(lambda t_h: case.sft) if ex.inner == "dirichlet" else None)
        Tex = ex(sol["r"][None, :], sol["t_h"][:, None] * H_TO_S)
        return sol, Tex, np.abs(sol["T"] - Tex)

    tests = {"A neumann mode 1": make_test([1], [20], "neumann", 0),
             "B neumann mode 3": make_test([3], [20], "neumann", 0),
             "C neumann mode 6": make_test([6], [20], "neumann", 0),
             "D neumann 1+3+6": make_test([1, 3, 6], [20, 10, 5], "neumann", 1),
             "E dirichlet mode 3": make_test([3], [20], "dirichlet", 0)}
    for name, (case, ex) in tests.items():
        err = run_fd(case, ex, 400, 2000)[2].max()
        print(f"  {name:20s} max err {err:.2e}  {'OK' if err < 0.2 else 'CHECK'}")
        assert err < 0.2
    case, ex = tests["D neumann 1+3+6"]
    order = lambda e: np.round([np.log(e[i] / e[i + 1]) / np.log(2) for i in range(len(e) - 1)], 2)
    e_sp = [run_fd(case, ex, n, 4000)[2].max() for n in (50, 100, 200, 400)]
    e_t = [run_fd(case, ex, 800, n)[2].max() for n in (25, 50, 100, 200)]
    e_j = [run_fd(case, ex, a, b)[2].max() for a, b in ((100, 250), (200, 500), (400, 1000), (800, 2000))]
    print("  observed order  space:", order(e_sp), " time:", order(e_t), " joint:", order(e_j))
    assert order(e_sp)[-1] > 1.9 and order(e_j)[-1] > 1.9
    case, ex = make_test([6], [20], "neumann", 0)
    w = load_well("SGIL"); sol = run_fd(case, ex, 400, 2000)[0]
    d = np.max(np.abs(np.interp(w.t_h, sol["t_h"], sol["bht"]) - ex(sol["r"][0], w.t_h * H_TO_S)))
    print(f"  dt-to-time mapping error at SGIL data times: {d:.2e} C")
    assert d < 1e-2 and np.isclose(sol["t_h"][-1], case.t_max_h)


for k, f in {1: step1, 2: step2, 3: step3, 4: step4, 5: step5}.items():
    if k in args.only:
        f()
print("\nall requested replication checks finished")

