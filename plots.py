import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import H_TO_S

DT = torch.float64


def plot_all(res, out_dir, tag):
    model, case, well, unit, hist = res["model"], res["case"], res["well"], res["unit"], res["hist"]
    tm = H_TO_S if unit == "s" else 1.0
    d = case.domain

    # Fig 3 analogue: total loss, linear (as in paper) and log axes, plus components
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ep = np.arange(len(hist["total"]))
    ax[0].plot(ep, hist["total"]); ax[0].set_title("Total loss (linear, as Fig. 3)")
    ax[1].semilogy(ep, hist["total"], label="total")
    for k in ("m", "p", "bc"):
        ax[2].semilogy(ep, np.maximum(hist[k], 1e-30), label=k)
    ax[1].set_title("Total loss (log)"); ax[2].set_title("Components (log)"); ax[2].legend()
    for a in ax: a.set_xlabel("epoch")
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(out_dir / "fig3_loss.png", dpi=130); plt.close(fig)

    # Fig 4 analogue: T(r_w, t) predicted vs data
    th = np.linspace(0, case.t_max_h, 400)
    with torch.no_grad():
        Tp = model(torch.full((400, 1), d.r_w, dtype=DT), torch.tensor(th * tm, dtype=DT).reshape(-1, 1)).numpy().ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th, Tp, label="Predicted"); ax.plot(well.t_h, well.bht, "r.", label="Actual")
    ax.axhline(case.sft, color="gray", ls=":", label=f"SFT BC = {case.sft:.1f}")
    ax.set_xlabel("t (h)"); ax.set_ylabel("T (°C)"); ax.set_title(tag); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "fig4_timeseries.png", dpi=130); plt.close(fig)

    # Fig 5 analogue: cross-plot
    with torch.no_grad():
        pred = model(torch.full((len(well.t_h), 1), d.r_w, dtype=DT),
                     torch.tensor(well.t_h * tm, dtype=DT).reshape(-1, 1)).numpy().ravel()
    lo, hi = min(well.bht.min(), pred.min()), max(well.bht.max(), pred.max())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8); ax.plot(well.bht, pred, "r.")
    ax.set_xlabel("Reference T (°C)"); ax.set_ylabel("Predicted T (°C)"); ax.set_title(tag)
    fig.tight_layout(); fig.savefig(out_dir / "fig5_crossplot.png", dpi=130); plt.close(fig)

    # Fig 6 PREVIEW only (full version is step 12): T vs r at several times
    rg = np.linspace(d.r_w, d.r_max, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        t_h = f * case.t_max_h
        with torch.no_grad():
            T = model(torch.tensor(rg, dtype=DT).reshape(-1, 1),
                      torch.full((300, 1), t_h * tm, dtype=DT)).numpy().ravel()
        ax.plot(rg, T, label=f"t = {t_h:.1f} h")
    ax.set_xlabel("r (m)"); ax.set_ylabel("T (°C)"); ax.set_title(tag + " (Fig. 6 preview)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig6_preview.png", dpi=130); plt.close(fig)

    