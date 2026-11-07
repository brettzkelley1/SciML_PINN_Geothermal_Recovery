"""make_replication_appendix.py: builds the appendix items that were only printed or never saved.
No training. Output: results/appendix/. Press Run, no flags."""
from pathlib import Path
from dataclasses import replace
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from config import H_TO_S
from data import all_wells, load_well, make_case, PAPER_TABLE1
from analytical import estimate_sft
from fd_solver import solve_fd
from exact_solutions import ExactSolution
from run_checks_replicate_NN import load12, field, OUT9, EPOCHS9
from train import build_model
from benchmark import predict, error_metrics

OUT = Path(__file__).resolve().parent / "results" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)


def save_table(path_png, title, header, rows):
    fig, ax = plt.subplots(figsize=(1.15 * len(header), 0.38 * (len(rows) + 2)))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path_png, dpi=200); plt.close(fig)


# ---------------- 1. step 3 table ----------------
def step3_table():
    W = all_wells()
    header = ["Well", "Asc SFT (ours)", "Asc SFT (paper)", "Hor SFT (ours)", "Hor SFT (paper)",
              "Asc |r| (ours)", "Asc 'R2' (paper)", "Hor |r| (ours)", "Hor 'R2' (paper)"]
    rows = []
    for n, (pa, ph, pra, prh) in PAPER_TABLE1.items():
        a, h = estimate_sft(W[n], "ascencio"), estimate_sft(W[n], "horner")
        rows.append([n, f"{a['sft']:.2f}", f"{pa:.2f}", f"{h['sft']:.2f}", f"{ph:.2f}",
                     f"{a['r']:.2f}", f"{pra:.2f}", f"{h['r']:.2f}", f"{prh:.2f}"])
    with open(OUT / "step3_table1_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    save_table(OUT / "step3_table1_comparison.png",
               "Static formation temperature (C): replication vs. Canilandi et al. Table 1", header, rows)

    vh = ["Well", "True T (C)", "Ascencio (C)", "Asc error %", "Horner (C)", "Hor error %"]
    vr = []
    for n, w in all_wells("val").items():
        a, h = estimate_sft(w, "ascencio")["sft"], estimate_sft(w, "horner")["sft"]
        vr.append([n, f"{w.tft:.1f}", f"{a:.2f}", f"{100 * (a - w.tft) / w.tft:+.1f}",
                   f"{h:.2f}", f"{100 * (h - w.tft) / w.tft:+.1f}"])
    with open(OUT / "step3_validation_wells.csv", "w", newline="") as f:
        csv.writer(f).writerows([vh] + vr)
    save_table(OUT / "step3_validation_wells.png",
               "Validation wells: estimated SFT vs. known true formation temperature", vh, vr)
    print("saved step 3 tables")


# ---------------- 2. FD convergence against an exact solution ----------------
def fd_convergence():
    base = make_case("SGIL")
    probe = ExactSolution(base, [1, 3, 6], [20, 10, 5], "neumann")
    case = replace(base, t_max_h=2.0 * probe.tau_s(1) / H_TO_S)
    ex = ExactSolution(case, [1, 3, 6], [20, 10, 5], "neumann")

    def err(n_r, n_t):
        sol = solve_fd(case, n_r=n_r, n_t=n_t, ic_fn=lambda r: ex(r, 0.0), wall_bc="neumann")
        return np.abs(sol["T"] - ex(sol["r"][None, :], sol["t_h"][:, None] * H_TO_S)).max()

    nr = [50, 100, 200, 400]; e_sp = [err(n, 4000) for n in nr]
    nt = [25, 50, 100, 200]; e_t = [err(800, n) for n in nt]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].loglog(nr, e_sp, "o-"); ax[0].loglog(nr, e_sp[0] * (np.array(nr) / nr[0]) ** -2.0, "k--", label="slope -2")
    ax[0].set_xlabel("radial nodes"); ax[0].set_ylabel("max error (C)"); ax[0].set_title("Spatial convergence"); ax[0].legend()
    ax[1].loglog(nt, e_t, "o-"); ax[1].loglog(nt, e_t[0] * (np.array(nt) / nt[0]) ** -2.0, "k--", label="slope -2")
    ax[1].set_xlabel("time steps"); ax[1].set_title("Temporal convergence"); ax[1].legend()
    for a in ax:
        a.xaxis.set_minor_formatter(NullFormatter())     # removes overlapping minor tick labels
    fig.suptitle("Finite-difference solver vs. exact eigenmode solution"); fig.tight_layout()
    fig.savefig(OUT / "step5_fd_convergence.png", dpi=150); plt.close(fig)
    print("saved FD convergence figure")


# ---------------- 3. Fig. 4 and 5 analogues from cached baseline models ----------------
def fig4_fig5(wells=("MXCO1", "MXCO2", "SGIL", "PHIL")):
    f4, a4 = plt.subplots(2, 2, figsize=(9, 6)); f5, a5 = plt.subplots(2, 2, figsize=(7, 6.5))
    for ax4, ax5, n in zip(a4.ravel(), a5.ravel(), wells):
        res = load12(n); case, well, mdl = res["case"], res["well"], res["model"]
        tt = np.linspace(0, case.t_max_h, 300)
        T = field(mdl, [case.domain.r_w], tt)[:, 0]
        pred = np.interp(well.t_h, tt, T)
        ax4.plot(tt, T, label="Predicted"); ax4.plot(well.t_h, well.bht, "r.", label="Actual")
        ax4.set_title(n); ax4.set_xlabel("t (h)"); ax4.set_ylabel("T (C)"); ax4.legend(fontsize=7)
        lo, hi = well.bht.min(), well.bht.max()
        ax5.plot([lo, hi], [lo, hi], "k-", lw=0.8); ax5.plot(well.bht, pred, "r.")
        ax5.set_title(n); ax5.set_xlabel("Reference T (C)"); ax5.set_ylabel("Predicted T (C)")
    f4.suptitle("Fig. 4 analogue: wall temperature vs. time"); f4.tight_layout(); f4.savefig(OUT / "fig4_analogue.png", dpi=150)
    f5.suptitle("Fig. 5 analogue: cross-plots"); f5.tight_layout(); f5.savefig(OUT / "fig5_analogue.png", dpi=150)
    plt.close("all"); print("saved Fig. 4 and 5 analogues")


# ---------------- 4. step 9 PINN vs FD figure from the cached model ----------------
def step9_figure():
    case, well = make_case("SGIL"), load_well("SGIL")
    sol = solve_fd(case, n_r=400, n_t=2000, wall_bc="neumann")
    model = build_model(case, "h", 0, 1e-3, EPOCHS9, True)
    model.load_state_dict(torch.load(OUT9 / "model.pt")); model = model.double().eval()
    Tp = predict(model, "h", sol["r"], sol["t_h"])
    m = error_metrics(Tp, sol, case)
    r, t = sol["r"], sol["t_h"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(t, sol["T"][:, 0], "k-", label="FD"); ax[0].plot(t, Tp[:, 0], "r--", label="PINN")
    ax[0].set_xlabel("t (h)"); ax[0].set_ylabel("T(r_w, t) (C)"); ax[0].set_title("Wall trace"); ax[0].legend()
    pc = ax[1].pcolormesh(r, t, np.abs(Tp - sol["T"]), shading="auto"); ax[1].set_xscale("log")
    ax[1].set_xlabel("r (m, log)"); ax[1].set_ylabel("t (h)"); ax[1].set_title("|PINN - FD| (C)"); fig.colorbar(pc, ax=ax[1])
    for f in (0.0, 0.02, 0.1, 0.5, 1.0):
        k = int(round(f * (len(t) - 1)))
        l, = ax[2].plot(r, sol["T"][k], "-", lw=1); ax[2].plot(r, Tp[k], "--", lw=1, color=l.get_color())
    ax[2].set_xscale("log"); ax[2].set_xlabel("r (m, log)"); ax[2].set_title("Profiles: FD solid, PINN dashed")
    for a in (ax[1], ax[2]):
        a.xaxis.set_minor_formatter(NullFormatter())     # removes overlapping minor tick labels
    fig.suptitle(f"Step 9 benchmark, SGIL: L_inf = {m['linf_C']:.3f} C ({m['linf_pct_dT']:.2f}% of dT)")
    fig.tight_layout(); fig.savefig(OUT / "step9_pinn_vs_fd.png", dpi=150); plt.close(fig)
    print(f"saved step 9 figure (L_inf {m['linf_C']:.3f} C)")


if __name__ == "__main__":
    step3_table(); fd_convergence(); fig4_fig5(); step9_figure()
    print(f"\nall appendix items in {OUT}")
    print("Copy these from results/step12/: fig6_all_wells.png, fig7_all_wells.png, fig8_maps.png")

