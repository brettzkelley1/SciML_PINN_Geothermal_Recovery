"""build_appendix_document.py (put in the Code folder, next to the two figure-generation scripts).
Assembles the figures made by replication_figure_generation.py and model_improvements/improvement_figure_generation.py
into one formatted Word document. It creates NO figures and trains nothing.
Needs: pip install python-docx pillow.   Output: results/Implementation_Appendices_v2.docx
The ORIGINAL paper figures are taken once from your existing appendices .docx (set OLD_DOCX below)."""
from pathlib import Path
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = Path(__file__).resolve().parent
REP = HERE / "results" / "figures_replication"
IMP = HERE / "model_improvements" / "results" / "figures_improved"
PAPER = HERE / "results" / "paper_figures"
OLD_DOCX = HERE / "Implementation_Appendices.docx"      # your current document (only used to pull the paper's figures)
OUT_DOCX = HERE / "results" / "Implementation_Appendices_v2.docx"

ORIG_CAP = "Original: Canilandi et al. (2025), IOP Conf. Ser.: Earth Environ. Sci. 1456, 012006 (CC BY 4.0)."
REPL_CAP = "Replication: normalized inputs and outputs, SGIL dt = 0 point dropped, 5,000 epochs, one seed."
IMPR_CAP = "Improved: learnable D, adaptive weights, IC and wall terms; IC width fixed at 0.5; 8,000 epochs, one seed."
missing = []


# ----------------------------- low-level helpers -----------------------------
def field(paragraph, instr, placeholder=""):
    for kind in ("begin", None, "separate", "text", "end"):
        r = paragraph.add_run()
        if kind == "text":
            r.text = placeholder
        elif kind is None:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r._r.append(it)
        else:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._r.append(fc)


def style_doc(doc):
    n = doc.styles["Normal"]; n.font.name = "Calibri"; n.font.size = Pt(10.5)
    n.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for name, size, before in (("Heading 1", 16, 14), ("Heading 2", 12.5, 10)):
        st = doc.styles[name]; st.font.name = "Calibri"; st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        rf = st.element.rPr.rFonts
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.space_before = Pt(before); st.paragraph_format.space_after = Pt(4)
    doc.settings.element.append(_el("w:updateFields", **{"w:val": "true"}))   # Word offers to refresh the contents list
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s, m, Inches(0.8))
    fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page "); r.font.size = Pt(9)
    field(fp, "PAGE", "1")


def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), v)
    return e


class Builder:
    def __init__(self):
        self.doc = Document(); style_doc(self.doc); self.landscape = False

    # usable page size in inches for the current (last) section
    def usable(self):
        s = self.doc.sections[-1]
        return ((s.page_width - s.left_margin - s.right_margin) / 914400,
                (s.page_height - s.top_margin - s.bottom_margin) / 914400)

    def start_page(self, landscape=False):
        """New page; adds a section break only when the orientation changes."""
        if landscape != self.landscape:
            s = self.doc.add_section(WD_SECTION.NEW_PAGE)
            s.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
            s.page_width, s.page_height = (Inches(11), Inches(8.5)) if landscape else (Inches(8.5), Inches(11))
            self.landscape = landscape
            return True
        return False

    def heading(self, text, level, new_page=False, landscape=None):
        landscape = self.landscape if landscape is None else landscape
        changed = self.start_page(landscape) if (new_page or landscape != self.landscape) else False
        h = self.doc.add_heading(text, level=level)
        if new_page and not changed:
            h.paragraph_format.page_break_before = True
        return h

    def text(self, s, italic=False, size=10, after=4, keep_next=False, color=None):
        p = self.doc.add_paragraph(); r = p.add_run(s); r.italic = italic; r.font.size = Pt(size)
        if color: r.font.color.rgb = color
        p.paragraph_format.space_after = Pt(after); p.paragraph_format.keep_with_next = keep_next
        return p

    def label(self, s, page_break=False):
        p = self.doc.add_paragraph(); r = p.add_run(s); r.bold = True; r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.page_break_before = page_break
        return p

    def figure(self, path, max_h, max_w=None, caption=None):
        uw, _ = self.usable(); max_w = min(max_w or uw, uw)
        path = Path(path)
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2); p.paragraph_format.keep_with_next = bool(caption)
        if not path.exists():
            missing.append(str(path))
            r = p.add_run(f"[figure not found: {path.name}]"); r.font.color.rgb = RGBColor(0xC0, 0, 0)
        else:
            w, h = Image.open(path).size
            p.add_run().add_picture(str(path), width=Inches(min(max_w, max_h * w / h)))
        if caption:
            c = self.text(caption, italic=True, size=8.5, after=8); c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # one appendix with Original / Replicated / Improved blocks
    def triplet(self, title, intro, orig, rep, imp, heights=(2.2, 2.15, 2.15), repl_cap=REPL_CAP, landscape=False):
        self.heading(title, 2, new_page=True, landscape=landscape)
        self.text(intro, size=10, after=6, keep_next=True)
        _, uh = self.usable()
        hs = [min(h, uh - 1.6) for h in (heights if not landscape else (uh - 1.9,) * 3)]
        for k, (lab, path, cap, h) in enumerate(zip(("Original (paper)", "Replicated", "Improved"),
                                                    (orig, rep, imp), (ORIG_CAP, repl_cap, IMPR_CAP), hs)):
            self.label(lab, page_break=(landscape and k > 0))
            self.figure(path, h, caption=cap)


def extract_originals():
    """Pull the paper's figures (Fig. 3-8) out of your existing appendices .docx, once."""
    names = ["fig3", "fig4", "fig5", "fig6", "fig7", "fig8"]
    if all((PAPER / f"{n}_paper.png").exists() for n in names):
        return
    if not OLD_DOCX.exists():
        print(f"!! {OLD_DOCX} not found and {PAPER} is incomplete: put your existing appendices .docx there "
              f"(or drop fig3_paper.png ... fig8_paper.png into {PAPER}).")
        return
    d = Document(str(OLD_DOCX)); blobs = []
    for t in d.tables:
        c = t.rows[0].cells
        if c[0].text.strip().lower() == "original":
            ids = c[1]._element.xpath(".//a:blip/@r:embed")
            if ids:
                blobs.append(d.part.related_parts[ids[0]].blob)
    if len(blobs) != 6:
        print(f"!! expected 6 'Original' figures in {OLD_DOCX.name}, found {len(blobs)}; check the file")
    PAPER.mkdir(parents=True, exist_ok=True)
    for n, b in zip(names, blobs):
        (PAPER / f"{n}_paper.png").write_bytes(b)
    print(f"extracted {len(blobs)} paper figures to {PAPER}")


def main():
    extract_originals()
    b = Builder(); doc = b.doc
    # ---- title and contents ----
    t = doc.add_paragraph(); r = t.add_run("Appendices"); r.bold = True; r.font.size = Pt(24)
    r.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
    b.text("Physics-informed neural network for geothermal well thermal recovery: replication of Canilandi et al. (2025) "
           "and improvements", italic=True, size=11, after=8)
    b.text("All replicated and improved results are single-seed. Overshoot means the predicted temperature exceeds the "
           "far-field static formation temperature, which a diffusion solution with that boundary value cannot do.",
           size=9.5, after=10)
    b.text("Contents", size=12, after=2).runs[0].bold = True
    p = doc.add_paragraph(); field(p, 'TOC \\o "1-2" \\h \\z \\u',
                                   "Right-click here and choose Update Field to build the contents list.")

    # ---- Section 1 ----
    b.heading("Section 1: Finite Difference Verification", 1, new_page=True)
    b.heading("Appendix 1: Finite Difference Model Convergence", 2)
    b.text("Maximum error of the finite-difference solver against an exact eigenmode solution; the dashed line is a "
           "second-order slope.", size=10, keep_next=True)
    b.figure(REP / "fd_convergence.png", 3.4, caption="Finite-difference solver verification (zero-flux wall, Dirichlet far field).")
    b.heading("Appendix 2: Finite Difference Benchmark PINN Model Comparison", 2)
    b.text("A PINN trained on the PDE, Gaussian initial condition, zero-flux wall and far-field condition (no data) compared "
           "with the finite-difference solution for SGIL.", size=10, keep_next=True)
    b.figure(REP / "pinn_vs_fd.png", 3.4, caption="Benchmark of the PINN against the finite-difference reference (15,000 epochs, one seed).")

    # ---- Section 2 ----
    b.heading("Section 2: Paper vs Replication vs \u201cImproved\u201d Model Results", 1, new_page=True)
    b.heading("Appendix 3a: SFT Estimates for the Paper and the Replication", 2)
    b.text("Static formation temperature (SFT) from the Ascencio and Horner regressions. The paper's \u201cR2\u201d column "
           "matches the Pearson correlation coefficient |r| of our fits, not the coefficient of determination.",
           size=10, keep_next=True)
    b.figure(REP / "sft_table1_comparison.png", 3.6, caption="Replication reproduces the paper's SFT estimates to two decimals.")
    b.heading("Appendix 3b: Validation Wells, SFT Estimates vs Known True Temperature", 2)
    b.figure(REP / "sft_validation_wells.png", 2.4,
             caption="Ascencio recovers the true temperature within 5% on three of four wells; KJ-21 is the exception.")

    b.triplet("Appendix 4: Loss Curves",
              "Training loss versus epoch. The paper's curve starts near 1.3e9; the replication does not reproduce that scale.",
              PAPER / "fig3_paper.png", REP / "fig3_replicated.png", IMP / "fig3_improved.png",
              heights=(2.1, 2.2, 2.2),
              repl_cap="Replication: normalized ('full') model, 15,000 epochs, one seed. Loss scale differs from the paper's.")
    b.triplet("Appendix 5: T vs t at the Wall",
              "Wall (bottom-hole) temperature against time; red points are the measured data.",
              PAPER / "fig4_paper.png", REP / "fig4_replicated.png", IMP / "fig4_improved.png",
              heights=(2.4, 2.1, 2.1))
    b.triplet("Appendix 6: Cross Plot",
              "Predicted against measured wall temperature; points on the line are perfect fits.",
              PAPER / "fig5_paper.png", REP / "fig5_replicated.png", IMP / "fig5_improved.png", heights=(2.1, 2.2, 2.2))
    b.triplet("Appendix 7: T vs r",
              "Radial temperature profiles at several times. Curves above the dotted far-field SFT line are overshoot.",
              PAPER / "fig6_paper.png", REP / "fig6_replicated.png", IMP / "fig6_improved.png", landscape=True)
    b.triplet("Appendix 8: T vs t at Several Radii",
              "Temperature history at several radii; black dots are the wall data.",
              PAPER / "fig7_paper.png", REP / "fig7_replicated.png", IMP / "fig7_improved.png", landscape=True)
    b.triplet("Appendix 9: 2-D Maps",
              "Radial temperature maps at selected times for SGIL, MXCO1 and SHBE. Values above the SFT are drawn in the top color.",
              PAPER / "fig8_paper.png", REP / "fig8_replicated.png", IMP / "fig8_improved.png", heights=(3.0, 1.7, 1.7))

    # ---- Section 3 ----
    b.heading("Section 3: Improvement Diagnostics and Validation", 1, new_page=True, landscape=False)
    b.heading("Appendix 10: Identifiability", 2)
    b.text("How well the data constrain the diffusivity D, and how D trades off against the unknown initial-condition width.",
           size=10, keep_next=True)
    b.figure(IMP / "identifiability.png", 2.9, caption="D is well constrained when the IC is known; a wrong IC width shifts the fitted D several-fold.")
    b.heading("Appendix 11: Synthetic Recovery of D", 2)
    b.text("Synthetic data with a known D, IC width fixed at the truth. v3 (8,000 epochs, D learning rate 1e-2) was used for the real wells.",
           size=10, keep_next=True)
    b.figure(IMP / "gate_table.png", 2.3, caption="Gate test: pass requires both truths within 15% and data RMSE below 0.3 C.")
    b.figure(IMP / "gate_D_trajectories.png", 2.9, caption="Learned D during training for each variant; dotted line is the truth.")
    b.heading("Appendix 12: Failed Joint D and Width Run", 2, new_page=True)
    b.text("Learning D and the IC width together on synthetic data did not recover either parameter; they ran to their bounds.",
           size=10, keep_next=True)
    b.figure(IMP / "joint_D_width_failed_table.png", 2.0, caption="Synthetic jobs of the first combined run.")
    b.figure(IMP / "joint_D_width_failed_trajectories.png", 2.9, caption="Trajectories of D and the IC width multiplier.")
    b.heading("Appendix 13: Baseline vs Improved", 2, new_page=True)
    b.text("Data fit and overshoot for all 14 wells. Learned D is an effective fitting value, not a measured diffusivity.",
           size=10, keep_next=True)
    b.figure(IMP / "baseline_vs_improved_bars.png", 2.9, caption="Improved data fit is worse on every well; overshoot is better on some wells and worse on others.")
    b.figure(IMP / "baseline_vs_improved_table.png", 4.6, caption="Single seed; overshoot is the maximum predicted temperature above the far-field SFT.")
    b.heading("Appendix 14: Held-out Validation Wells", 2, new_page=True)
    b.text("Wall temperature at the end of the window against the known true formation temperature. The far-field condition is "
           "our own SFT estimate, so these errors largely repeat the SFT error.", size=10, keep_next=True)
    b.figure(IMP / "validation_wells_table.png", 2.4, caption="Not an independent prediction test.")
    b.heading("Appendix 15: Far-Field Radius Sensitivity", 2)
    b.text("Change in the finite-difference wall trace when the far-field radius moves from 6.5 m.", size=10, keep_next=True)
    b.figure(IMP / "rmax_sensitivity_table.png", 4.4, caption="The 6.5 m boundary does not affect any well beyond 0.5 C.")

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))
    print(f"saved {OUT_DOCX}")
    if missing:
        print(f"\n{len(missing)} figure(s) not found (shown as red placeholders in the document):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()