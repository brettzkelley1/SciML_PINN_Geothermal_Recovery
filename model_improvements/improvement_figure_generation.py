"""model_improvements/improvement_figure_generation.py: ALL improvement-section figures and tables. No training.
Reads cached results in model_improvements/results (step12d, step12c, step12_ic) and the baseline models.
Output: model_improvements/results/figures_improved/. Press Run, no flags."""
import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import textwrap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from step10_learnable_D import D_NOM, WELL, build_inverse_model, make_synthetic_well, nominal_ic, fit_logD, wall_trace
from step12c_gate import VARIANTS, TRUTH, OUT as OUT12C
from step12d_real_wells import well_case, OUT as OUT12D
from run_checks_replicate_NN import load12, field
from config import Domain, RockProps, H_TO_S
from data import all_wells, load_well, make_case
from fd_solver import solve_fd

OUT = HERE / "results" / "figures_improved"
OUT12IC = HERE / "results" / "step12_ic"
dom, D_ROCK = Domain(), RockProps().D
TAG = "improved pipeline (learnable D, adaptive weights, IC + wall terms)"
RG = np.linspace(dom.r_w, dom.r_max, 200)


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


def load_improved(name):
    """Returns (case, well, model) for the step 12d improved model."""
    case, well = well_case(name)
    model = build_inverse_model(case, D_NOM)
    model.load_state_dict(torch.load(OUT12D / name / "model.pt"))
    return case, well, model.double().eval()


def wall_rmse(case, well, mdl):
    return float(np.sqrt(np.mean((field(mdl, [dom.r_w], well.t_h)[:, 0] - well.bht) ** 2)))


def overshoot(case, mdl):
    return float(field(mdl, RG, np.linspace(0, case.t_max_h, 60)).max() - case.sft)


# ---------------- improved versions of the paper figures ----------------
def fig3_improved():
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for i, n in enumerate(("SGIL", "MXCO1")):
        h = np.load(OUT12D / n / "history.npz")
        ax[i, 0].semilogy(h["total"]); ax[i, 0].set_title(f"{n}: weighted total loss")
        for k in ("m", "p", "bc", "ic", "wall"):
            ax[i, 1].semilogy(np.maximum(h[k], 1e-30), lw=0.8, label=k)
        ax[i, 1].set_title(f"{n}: unweighted components"); ax[i, 1].legend(fontsize=7)
        ax[i, 2].plot(h["D"] / D_NOM); ax[i, 2].axhline(1, color="k", ls=":"); ax[i, 2].set_title(f"{n}: learned D / D_nom")
        for a in ax[i]: a.set_xlabel("epoch")
    fig.suptitle("Fig. 3 analogue, " + TAG); fig.tight_layout(); fig.savefig(OUT / "fig3_improved.png", dpi=150); plt.close(fig)


def fig4_fig5_improved(wells=("MXCO1", "MXCO2", "SGIL", "PHIL")):
    f4, a4 = plt.subplots(2, 2, figsize=(9, 6)); f5, a5 = plt.subplots(2, 2, figsize=(7, 6.5))
    for x4, x5, n in zip(a4.ravel(), a5.ravel(), wells):
        case, well, mdl = load_improved(n)
        tt = np.linspace(0, case.t_max_h, 300); T = field(mdl, [dom.r_w], tt)[:, 0]
        pred = field(mdl, [dom.r_w], well.t_h)[:, 0]
        x4.plot(tt, T, label="Predicted"); x4.plot(well.t_h, well.bht, "r.", label="Actual")
        x4.set_title(n); x4.set_xlabel("t (h)"); x4.set_ylabel("T (C)"); x4.legend(fontsize=7)
        lo, hi = well.bht.min(), well.bht.max()
        x5.plot([lo, hi], [lo, hi], "k-", lw=0.8); x5.plot(well.bht, pred, "r.")
        x5.set_title(n); x5.set_xlabel("Reference T (C)"); x5.set_ylabel("Predicted T (C)")
    f4.suptitle("Fig. 4 analogue, " + TAG, fontsize=10); f4.tight_layout(); f4.savefig(OUT / "fig4_improved.png", dpi=150)
    f5.suptitle("Fig. 5 analogue, " + TAG, fontsize=10); f5.tight_layout(); f5.savefig(OUT / "fig5_improved.png", dpi=150)
    plt.close("all")


def fig6_fig7_fig8_improved():
    names = list(all_wells().keys()); runs = {n: load_improved(n) for n in names}
    ncol, nrow = 4, int(np.ceil(len(names) / 4))
    f6, a6 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow)); a6 = a6.ravel()
    f7, a7 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow)); a7 = a7.ravel()
    for i, n in enumerate(names):
        case, well, mdl = runs[n]
        for f in (0.0, 0.1, 0.25, 0.5, 1.0):
            a6[i].plot(RG, field(mdl, RG, [f * case.t_max_h])[0], lw=1, label=f"{f * case.t_max_h:.0f} h")
        a6[i].axhline(case.sft, color="gray", ls=":", lw=0.8)
        a6[i].set_title(n); a6[i].set_xlabel("r (m)"); a6[i].set_ylabel("T (C)"); a6[i].legend(fontsize=6)
        tt = np.linspace(0, case.t_max_h, 200)
        for r0 in (dom.r_w, 0.5, 1.58, 3.69, dom.r_max):
            a7[i].plot(tt, field(mdl, [r0], tt)[:, 0], lw=1, label=f"r={r0:.2f} m")
        a7[i].plot(well.t_h, well.bht, "k.", ms=4)
        a7[i].set_title(n); a7[i].set_xlabel("t (h)"); a7[i].set_ylabel("T (C)"); a7[i].legend(fontsize=6)
    for a in list(a6[len(names):]) + list(a7[len(names):]): a.axis("off")
    f6.suptitle("Fig. 6 analogue, " + TAG + " (dotted = far-field SFT)"); f6.tight_layout(); f6.savefig(OUT / "fig6_improved.png", dpi=120)
    f7.suptitle("Fig. 7 analogue, " + TAG); f7.tight_layout(); f7.savefig(OUT / "fig7_improved.png", dpi=120); plt.close("all")

    sel, fr = ["SGIL", "MXCO1", "SHBE"], (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    xy = np.linspace(-5, 5, 200); X, Y = np.meshgrid(xy, xy); Rr = np.sqrt(X ** 2 + Y ** 2)
    fig, axs = plt.subplots(len(sel), len(fr), figsize=(2.7 * len(fr) + 1.0, 2.7 * len(sel)), constrained_layout=True)
    for i, n in enumerate(sel):
        case, well, mdl = runs[n]
        vmin = float(min(field(mdl, RG, [f * case.t_max_h]).min() for f in fr))
        levels = np.linspace(vmin, case.sft, 16); cs = None
        for j, f in enumerate(fr):
            Tm = np.interp(Rr, RG, field(mdl, RG, [f * case.t_max_h])[0]); Tm[Rr < dom.r_w] = np.nan
            cs = axs[i, j].contourf(X, Y, Tm, levels=levels, extend="max", cmap="plasma")
            axs[i, j].set_title(f"{n} {f * case.t_max_h:.0f} h", fontsize=8); axs[i, j].set_aspect("equal")
        fig.colorbar(cs, ax=axs[i, :].tolist(), shrink=0.85, label="T (C)")
    fig.suptitle("Fig. 8 analogue, " + TAG); fig.savefig(OUT / "fig8_improved.png", dpi=120); plt.close(fig)


# ---------------- baseline vs improved ----------------
def baseline_vs_improved():
    names = list(all_wells().keys()); rows, br, ir, bo, io = [], [], [], [], []
    for n in names:
        b = load12(n); case, well, mi = load_improved(n)
        h = np.load(OUT12D / n / "history.npz")
        br.append(wall_rmse(case, b["well"], b["model"])); ir.append(wall_rmse(case, well, mi))
        bo.append(overshoot(case, b["model"])); io.append(overshoot(case, mi))
        rows.append([n, f"{br[-1]:.2f}", f"{ir[-1]:.2f}", f"{bo[-1]:+.2f}", f"{io[-1]:+.2f}", f"{h['D'][-1] / D_NOM:.2f}"])
    save_table(OUT / "baseline_vs_improved_table.png",
               "Baseline vs. improved pipeline (IC width fixed at 0.5 for the improved model)",
               ["Well", "Baseline RMSE (C)", "Improved RMSE (C)", "Baseline overshoot (C)", "Improved overshoot (C)", "Learned D / D_nom"],
               rows, note="Overshoot = max predicted T above the far-field SFT (my check). Learned D is an effective value, not a measurement. One seed.")
    x, w = np.arange(len(names)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].bar(x - w / 2, br, w, label="baseline"); ax[0].bar(x + w / 2, ir, w, label="improved")
    ax[0].set_ylabel("wall-data RMSE (C)"); ax[0].set_title("Data fit")
    ax[1].bar(x - w / 2, bo, w, label="baseline"); ax[1].bar(x + w / 2, io, w, label="improved")
    ax[1].set_ylabel("max T above far-field SFT (C)"); ax[1].set_title("Overshoot (lower is better; my check)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=60); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "baseline_vs_improved_bars.png", dpi=150); plt.close(fig)


# ---------------- identifiability (steps 10, 10b) ----------------
def identifiability():
    case, template = make_case(WELL), load_well(WELL, drop_dt0=True)
    ic0 = nominal_ic(case); t_obs = template.t_h
    ref = wall_trace(case, D_NOM, t_obs, ic0, 400, 2000)
    mult = np.array([0.25, 0.5, 0.8, 1.25, 2.0, 4.0])
    rm = [float(np.sqrt(np.mean((wall_trace(case, D_NOM * m, t_obs, ic0, 400, 2000) - ref) ** 2))) for m in mult]
    ems = (0.5, 1.0, 2.0); conf = []
    for em in ems:
        syn, _ = make_synthetic_well(case, template, D_NOM, 0.0, 0, nominal_ic(case, em))
        conf.append(fit_logD(case, syn.t_h, syn.bht, nominal_ic(case, 1.0))[0] / D_NOM)
    w_ = (0.25, 0.5, 1.0)
    real = [fit_logD(case, template.t_h, template.bht, nominal_ic(case, e))[0] / D_NOM for e in w_]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogx(mult, rm, "o-"); ax[0].set_xlabel("D / D_true"); ax[0].set_ylabel("wall-trace RMSE vs true trace (C)")
    ax[0].set_xticks([0.25, 0.5, 1, 2, 4]); ax[0].set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax[0].xaxis.set_minor_formatter(NullFormatter()); ax[0].set_title("D is well constrained by the SGIL wall trace\n(FD only)")
    ax[1].bar([f"{e}x" for e in ems], conf, color=["#c44", "#4a4", "#c44"]); ax[1].axhline(1, color="k", ls=":")
    ax[1].set_xlabel("true IC width multiplier"); ax[1].set_ylabel("fitted D / true D")
    ax[1].set_title("IC width and D are confounded\n(D fitted with the nominal IC width)")
    ax[2].plot(w_, real, "o-"); ax[2].set_yscale("log"); ax[2].set_xlabel("assumed IC width multiplier")
    ax[2].set_ylabel("fitted D / D_nom"); ax[2].set_title("Real SGIL data: fitted D vs. assumed IC width")
    ax[2].yaxis.set_minor_formatter(NullFormatter())
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle("Identifiability of the diffusivity (steps 10 and 10b)"); fig.tight_layout()
    fig.savefig(OUT / "identifiability.png", dpi=170); plt.close(fig)


# ---------------- synthetic gate (step 12c) and failed joint run (step 12) ----------------
def gate():
    rows, ok_any = [], False
    for v, (ep, lr, share) in VARIANTS.items():
        ms = {}
        for t in TRUTH:
            p = OUT12C / f"{v}_{t}" / "metrics.json"
            if p.exists(): ms[t] = json.loads(p.read_text())
        if len(ms) < 2:
            continue
        ok_any = True
        rec = {t: ms[t]["D_ratio"] / TRUTH[t] for t in TRUTH}
        ok = all(abs(rec[t] - 1) < 0.15 and ms[t]["rmse"] < 0.3 for t in TRUTH)
        rows.append([v, str(ep), f"{lr:g}", f"{share.get('ic', 1.0):g}", f"{rec['A']:.3f}", f"{rec['B']:.3f}",
                     f"{max(ms['A']['rmse'], ms['B']['rmse']):.2f}", "PASS" if ok else "FAIL"])
    if ok_any:
        save_table(OUT / "gate_table.png", "Synthetic recovery of D with the IC width fixed at truth (step 12c)",
                   ["Variant", "Epochs", "D learning rate", "IC share", "A: D_hat/D_true (true 0.5x)", "B: D_hat/D_true (true 2x)",
                    "Worst data RMSE (C)", "Verdict"], rows,
                   note="Pass = both truths within 15% and data RMSE < 0.3 C. Real-well runs used the 8000-epoch, lr_D 1e-2 variant.")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, (t, tv) in enumerate(TRUTH.items()):
            for p in sorted(OUT12C.glob(f"*_{t}/history.npz")):
                ax[i].plot(np.load(p)["D"] / D_NOM, lw=1, label=p.parent.name.rsplit("_", 1)[0])
            ax[i].axhline(tv, color="k", ls=":"); ax[i].set_title(f"synthetic truth D = {tv}x nominal")
            ax[i].set_xlabel("epoch"); ax[i].set_ylabel("D / D_nom"); ax[i].legend(fontsize=6)
        fig.suptitle("Step 12c: recovery of D on synthetic data (IC width fixed at truth)")
        fig.tight_layout(); fig.savefig(OUT / "gate_D_trajectories.png", dpi=150); plt.close(fig)
    else:
        print("  no step 12c results found; skipping gate figures")


def joint_failed():
    ms = {n: json.loads((OUT12IC / n / "metrics.json").read_text()) for n in ("syn_A", "syn_B") if (OUT12IC / n / "metrics.json").exists()}
    if len(ms) < 2:
        print("  no step 12 (joint D + width) results found; skipping"); return
    rows = [[n, f"{m['D_true_ratio']:.2f}", f"{m['D_ratio_nom']:.2f}", f"{m['ell_true']:.2f}", f"{m['ell_mult']:.2f}", f"{m['rmse']:.2f}"]
            for n, m in ms.items()]
    wells_ = [json.loads(p.read_text()) for p in OUT12IC.glob("*/metrics.json") if not p.parent.name.startswith("syn")]
    n_bound = sum(m["ell_mult"] > 3.9 for m in wells_)
    save_table(OUT / "joint_D_width_failed_table.png", "Joint learning of D and IC width on synthetic data (first combined run)",
               ["Job", "True D / D_nom", "Learned D / D_nom", "True IC width", "Learned IC width", "Data RMSE (C)"], rows,
               note=f"On the real wells of that run, {n_bound} of {len(wells_)} learned an IC width at the 4.0 bound; D ranged "
                    f"{min(m['D_ratio_nom'] for m in wells_):.0f}x-{max(m['D_ratio_nom'] for m in wells_):.0f}x nominal. Results discarded.")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in ms:
        h = np.load(OUT12IC / n / "history.npz"); ax[0].plot(h["D"] / D_NOM, label=n); ax[1].plot(h["ell"], label=n)
    ax[0].set_ylabel("D / D_nom"); ax[1].set_ylabel("IC width multiplier")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    fig.suptitle("Joint learning of D and IC width: parameters run to their bounds"); fig.tight_layout()
    fig.savefig(OUT / "joint_D_width_failed_trajectories.png", dpi=150); plt.close(fig)


# ---------------- held-out validation and r_max sensitivity (step 13) ----------------
def validation():
    rows = []
    for n, w in all_wells("val").items():
        b = load12(n); case, _, mi = load_improved(n)
        tb = field(b["model"], [dom.r_w], [case.t_max_h])[0, 0]; ti = field(mi, [dom.r_w], [case.t_max_h])[0, 0]
        e = lambda v: 100 * (v - w.tft) / w.tft
        rows.append([n, f"{w.tft:.1f}", f"{case.sft:.2f}", f"{e(case.sft):+.1f}", f"{tb:.2f}", f"{e(tb):+.1f}", f"{ti:.2f}", f"{e(ti):+.1f}"])
    save_table(OUT / "validation_wells_table.png", "Held-out wells: wall temperature at end of window vs. known true formation temperature",
               ["Well", "True T (C)", "SFT used (C)", "SFT error %", "Baseline T_end (C)", "Baseline err %", "Improved T_end (C)", "Improved err %"],
               rows, note="The far-field BC is our own SFT estimate, so these errors largely repeat the SFT error; not an independent prediction test.")


def rmax_sensitivity():
    rows = []
    for n in all_wells():
        case, _ = well_case(n); t = np.linspace(0, case.t_max_h, 41)[1:]

        def trace(rm):
            c = replace(case, domain=replace(case.domain, r_max=rm))
            s = solve_fd(c, n_r=200, n_t=1000)
            return np.interp(t, s["t_h"], s["bht"])
        ref = trace(6.5)
        d = [np.abs(trace(rm) - ref).max() for rm in (3.0, 10.0, 15.0)]
        rows.append([n, f"{case.t_max_h:.0f}", f"{np.sqrt(D_ROCK * case.t_max_h * H_TO_S):.2f}", f"{d[0]:.3f}", f"{d[1]:.3f}", f"{d[2]:.3f}"])
    save_table(OUT / "rmax_sensitivity_table.png", "Far-field radius sensitivity (FD): max change in wall trace vs. r_max = 6.5 m (C)",
               ["Well", "Window (h)", "sqrt(D t) (m)", "r_max = 3 m", "r_max = 10 m", "r_max = 15 m"], rows,
               note="Default Gaussian IC and zero-flux wall.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig3_improved, fig4_fig5_improved, fig6_fig7_fig8_improved, baseline_vs_improved,
              identifiability, gate, joint_failed, validation, rmax_sensitivity):
        print(f"running {f.__name__} ...", flush=True); f()
    print(f"\nall improvement figures in {OUT}")


