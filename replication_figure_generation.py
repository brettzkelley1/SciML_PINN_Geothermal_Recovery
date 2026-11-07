"""replication_figure_generation.py: ALL replication-section figures and tables. No training.
Reads cached models/histories (results/step12, results/step9_replicate, results/step8_final).
Output: results/figures_replication/. Press Run, no flags."""
from pathlib import Path
from dataclasses import replace
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from config import H_TO_S, Domain
from data import all_wells, load_well, make_case, PAPER_TABLE1
from analytical import estimate_sft
from fd_solver import solve_fd
from exact_solutions import ExactSolution
from run_checks_replicate_NN import load12, field, OUT9, EPOCHS9
from train import build_model
from benchmark import predict, error_metrics
import textwrap

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "figures_replication"
dom = Domain()


def save_table(path, title, header, rows, note=None):
    header = [textwrap.fill(h, 14) for h in header]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(header)), 0.36 * (len(rows) + 4) + (0.4 if note else 0)))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.3)
    t.auto_set_column_width(list(range(len(header))))
    for j in range(len(header)):
        t[0, j].set_facecolor("#e8e8e8"); t[0, j].set_text_props(weight="bold")
        t[0, j].set_height(t[0, j].get_height() * 2.2)
    ax.set_title(title, fontsize=10, pad=6)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------- Appendix: SFT tables (step 3) ----------------
def sft_tables():
    W = all_wells()
    header = ["Well", "Asc SFT (ours)", "Asc SFT (paper)", "Hor SFT (ours)", "Hor SFT (paper)",
              "Asc |r| (ours)", "Asc 'R2' (paper)", "Hor |r| (ours)", "Hor 'R2' (paper)"]
    rows = []
    for n, (pa, ph, pra, prh) in PAPER_TABLE1.items():
        a, h = estimate_sft(W[n], "ascencio"), estimate_sft(W[n], "horner")
        rows.append([n, f"{a['sft']:.2f}", f"{pa:.2f}", f"{h['sft']:.2f}", f"{ph:.2f}",
                     f"{a['r']:.2f}", f"{pra:.2f}", f"{h['r']:.2f}", f"{prh:.2f}"])
    with open(OUT / "sft_table1_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    save_table(OUT / "sft_table1_comparison.png", "Static formation temperature (C): replication vs. Canilandi et al. Table 1",
               header, rows)
    vh = ["Well", "True T (C)", "Ascencio (C)", "Asc error %", "Horner (C)", "Hor error %"]
    vr = []
    for n, w in all_wells("val").items():
        a, h = estimate_sft(w, "ascencio")["sft"], estimate_sft(w, "horner")["sft"]
        vr.append([n, f"{w.tft:.1f}", f"{a:.2f}", f"{100 * (a - w.tft) / w.tft:+.1f}", f"{h:.2f}", f"{100 * (h - w.tft) / w.tft:+.1f}"])
    with open(OUT / "sft_validation_wells.csv", "w", newline="") as f:
        csv.writer(f).writerows([vh] + vr)
    save_table(OUT / "sft_validation_wells.png", "Validation wells: estimated SFT vs. known true formation temperature", vh, vr)
    print("saved SFT tables")


# ---------------- Section 1: FD verification (steps 5, 9) ----------------
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
        a.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Finite-difference solver vs. exact eigenmode solution"); fig.tight_layout()
    fig.savefig(OUT / "fd_convergence.png", dpi=150); plt.close(fig)
    print("saved FD convergence")


def pinn_vs_fd():
    case = make_case("SGIL")
    sol = solve_fd(case, n_r=400, n_t=2000, wall_bc="neumann")
    model = build_model(case, "h", 0, 1e-3, EPOCHS9, True)
    model.load_state_dict(torch.load(OUT9 / "model.pt")); model = model.double().eval()
    Tp = predict(model, "h", sol["r"], sol["t_h"])
    m = error_metrics(Tp, sol, case); r, t = sol["r"], sol["t_h"]
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
        a.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(f"Step 9 benchmark, SGIL: L_inf = {m['linf_C']:.3f} C ({m['linf_pct_dT']:.2f}% of dT)")
    fig.tight_layout(); fig.savefig(OUT / "pinn_vs_fd.png", dpi=150); plt.close(fig)
    print(f"saved PINN vs FD (L_inf {m['linf_C']:.3f} C)")


# ---------------- Section 2: replicated figures ----------------
def fig3_replicated():
    root = HERE / "results" / "step8_final"
    fig, ax = plt.subplots(2, 3, figsize=(15, 7)); n_ok = 0
    for i, n in enumerate(("SGIL", "MXCO1")):
        files = sorted((root / f"{n}_full").glob("*/history.npz"))
        if not files:
            print(f"  no step 8 history for {n} under {root}; skipping that row"); continue
        h = np.load(files[0]); n_ok += 1; ep = np.arange(len(h["total"]))
        ax[i, 0].plot(ep, h["total"]); ax[i, 0].set_title(f"{n}: total loss (linear, as paper Fig. 3)")
        ax[i, 1].semilogy(ep, h["total"]); ax[i, 1].set_title(f"{n}: total loss (log)")
        for k in ("m", "p", "bc"):
            ax[i, 2].semilogy(ep, np.maximum(h[k], 1e-30), lw=0.8, label=k)
        ax[i, 2].set_title(f"{n}: components"); ax[i, 2].legend(fontsize=7)
        for a in ax[i]: a.set_xlabel("epoch")
    fig.suptitle("Fig. 3 analogue, replication ('full' normalised mode, 15,000 epochs)")
    fig.tight_layout()
    if n_ok:
        fig.savefig(OUT / "fig3_replicated.png", dpi=150); print("saved Fig. 3 analogue")
    plt.close(fig)


def fig4_fig5(wells=("MXCO1", "MXCO2", "SGIL", "PHIL")):
    f4, a4 = plt.subplots(2, 2, figsize=(9, 6)); f5, a5 = plt.subplots(2, 2, figsize=(7, 6.5))
    for x4, x5, n in zip(a4.ravel(), a5.ravel(), wells):
        res = load12(n); case, well, mdl = res["case"], res["well"], res["model"]
        tt = np.linspace(0, case.t_max_h, 300); T = field(mdl, [dom.r_w], tt)[:, 0]
        pred = field(mdl, [dom.r_w], well.t_h)[:, 0]
        x4.plot(tt, T, label="Predicted"); x4.plot(well.t_h, well.bht, "r.", label="Actual")
        x4.set_title(n); x4.set_xlabel("t (h)"); x4.set_ylabel("T (C)"); x4.legend(fontsize=7)
        lo, hi = well.bht.min(), well.bht.max()
        x5.plot([lo, hi], [lo, hi], "k-", lw=0.8); x5.plot(well.bht, pred, "r.")
        x5.set_title(n); x5.set_xlabel("Reference T (C)"); x5.set_ylabel("Predicted T (C)")
    f4.suptitle("Fig. 4 analogue: wall temperature vs. time (replication)"); f4.tight_layout()
    f4.savefig(OUT / "fig4_replicated.png", dpi=150)
    f5.suptitle("Fig. 5 analogue: cross-plots (replication)"); f5.tight_layout()
    f5.savefig(OUT / "fig5_replicated.png", dpi=150); plt.close("all"); print("saved Fig. 4 and 5 analogues")


def fig6_fig7_fig8_and_summary():
    names = list(all_wells().keys()); runs = {n: load12(n) for n in names}
    rg = np.linspace(dom.r_w, dom.r_max, 200)
    ncol, nrow = 4, int(np.ceil(len(names) / 4))
    f6, a6 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow)); a6 = a6.ravel()
    f7, a7 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow)); a7 = a7.ravel()
    rows = []
    for i, n in enumerate(names):
        r = runs[n]; case, well, mdl = r["case"], r["well"], r["model"]
        for f in (0.0, 0.1, 0.25, 0.5, 1.0):
            a6[i].plot(rg, field(mdl, rg, [f * case.t_max_h])[0], lw=1, label=f"{f * case.t_max_h:.0f} h")
        a6[i].axhline(case.sft, color="gray", ls=":", lw=0.8)
        a6[i].set_title(n); a6[i].set_xlabel("r (m)"); a6[i].set_ylabel("T (C)"); a6[i].legend(fontsize=6)
        tt = np.linspace(0, case.t_max_h, 200)
        for r0 in (dom.r_w, 0.5, 1.58, 3.69, dom.r_max):
            a7[i].plot(tt, field(mdl, [r0], tt)[:, 0], lw=1, label=f"r={r0:.2f} m")
        a7[i].plot(well.t_h, well.bht, "k.", ms=4)
        a7[i].set_title(n); a7[i].set_xlabel("t (h)"); a7[i].set_ylabel("T (C)"); a7[i].legend(fontsize=6)
        pred = field(mdl, [dom.r_w], well.t_h)[:, 0]
        over = float(field(mdl, rg, np.linspace(0, case.t_max_h, 60)).max() - case.sft)
        rows.append([n, load_well(n).group, f"{np.sqrt(np.mean((pred - well.bht) ** 2)):.2f}",
                     f"{abs(np.corrcoef(pred, well.bht)[0, 1]):.3f}", f"{over:+.2f}", f"{case.sft:.1f}"])
    for a in list(a6[len(names):]) + list(a7[len(names):]): a.axis("off")
    f6.suptitle("Fig. 6 analogue: T vs r at each timestep (replication; dotted = far-field SFT)"); f6.tight_layout()
    f6.savefig(OUT / "fig6_replicated.png", dpi=120)
    f7.suptitle("Fig. 7 analogue: T vs t at each radius (replication; black dots = BHT data)"); f7.tight_layout()
    f7.savefig(OUT / "fig7_replicated.png", dpi=120); plt.close("all")

    sel, fr = ["SGIL", "MXCO1", "SHBE"], (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    xy = np.linspace(-5, 5, 200); X, Y = np.meshgrid(xy, xy); Rr = np.sqrt(X ** 2 + Y ** 2)
    fig, axs = plt.subplots(len(sel), len(fr), figsize=(2.7 * len(fr) + 1.0, 2.7 * len(sel)), constrained_layout=True)
    for i, n in enumerate(sel):
        case, mdl = runs[n]["case"], runs[n]["model"]
        vmin = float(min(field(mdl, rg, [f * case.t_max_h]).min() for f in fr))
        levels = np.linspace(vmin, case.sft, 16); cs = None
        for j, f in enumerate(fr):
            Tm = np.interp(Rr, rg, field(mdl, rg, [f * case.t_max_h])[0]); Tm[Rr < dom.r_w] = np.nan
            cs = axs[i, j].contourf(X, Y, Tm, levels=levels, extend="max", cmap="plasma")
            axs[i, j].set_title(f"{n} {f * case.t_max_h:.0f} h", fontsize=8); axs[i, j].set_aspect("equal")
        fig.colorbar(cs, ax=axs[i, :].tolist(), shrink=0.85, label="T (C)")
    fig.suptitle("Fig. 8 analogue: radial maps (replication; values above SFT drawn in the top color)")
    fig.savefig(OUT / "fig8_replicated.png", dpi=120); plt.close(fig)
    save_table(OUT / "replication_summary_table.png", "Replication baseline: data fit and overshoot, all wells",
               ["Well", "Group", "Wall RMSE (C)", "|r|", "Overshoot (C)", "Far-field SFT (C)"], rows,
               note="Overshoot = max predicted T above the far-field SFT (my check, not the paper's). 5,000 epochs, one seed.")
    print("saved Fig. 6, 7, 8 analogues and summary table")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sft_tables(); fd_convergence(); pinn_vs_fd(); fig3_replicated(); fig4_fig5(); fig6_fig7_fig8_and_summary()
    print(f"\nall replication figures in {OUT}")

