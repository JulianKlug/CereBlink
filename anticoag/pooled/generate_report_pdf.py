"""
Generate a PDF report for the propensity analysis results.

Combines narrative text, tables, and figures into a multi-page PDF
using matplotlib's PdfPages backend.

Output: pooled/artifacts/propensity_analysis_report.pdf
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

# Page geometry
PAGE_W, PAGE_H = 8.5, 11  # US letter in inches
MARGIN = 0.75


def new_page(pdf, title=None):
    """Create a new page with optional title. Returns (fig, ax)."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")
    if title:
        ax.text(PAGE_W / 2, PAGE_H - MARGIN + 0.15, title,
                ha="center", va="top", fontsize=14, fontweight="bold")
    return fig, ax


def draw_text_block(ax, x, y, text, fontsize=9, lineheight=0.18, max_width=90,
                    fontweight="normal", color="black"):
    """Draw wrapped text block. Returns y position after last line."""
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y -= lineheight * 0.6
            continue
        lines = textwrap.wrap(paragraph, width=max_width) or [""]
        for line in lines:
            if y < MARGIN:
                return y  # ran out of page
            ax.text(x, y, line, fontsize=fontsize, va="top",
                    fontweight=fontweight, color=color,
                    fontfamily="sans-serif")
            y -= lineheight
    return y


def draw_table(ax, df, x, y, fontsize=8, col_widths=None, row_height=0.17,
               header_color="#2c3e50", header_text_color="white",
               stripe_color="#f0f4f8"):
    """Draw a DataFrame as a table on the axes. Returns y after table."""
    n_cols = len(df.columns)
    n_rows = len(df)

    if col_widths is None:
        total_w = PAGE_W - 2 * MARGIN
        col_widths = [total_w / n_cols] * n_cols

    total_w = sum(col_widths)

    # Header background
    from matplotlib.patches import FancyBboxPatch, Rectangle
    header_rect = Rectangle((x, y - row_height), total_w, row_height,
                             facecolor=header_color, edgecolor="none")
    ax.add_patch(header_rect)

    # Header text
    cx = x
    for j, col in enumerate(df.columns):
        ax.text(cx + col_widths[j] / 2, y - row_height / 2, col,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=header_text_color,
                fontfamily="sans-serif")
        cx += col_widths[j]

    y -= row_height

    # Rows
    for i in range(n_rows):
        if y - row_height < MARGIN:
            break
        # Stripe
        if i % 2 == 0:
            stripe = Rectangle((x, y - row_height), total_w, row_height,
                                facecolor=stripe_color, edgecolor="none")
            ax.add_patch(stripe)

        cx = x
        for j, col in enumerate(df.columns):
            val = str(df.iloc[i, j])
            ha = "left" if j == 0 else "center"
            offset = 0.05 if j == 0 else col_widths[j] / 2
            ax.text(cx + offset, y - row_height / 2, val,
                    ha=ha, va="center", fontsize=fontsize,
                    fontfamily="sans-serif")
            cx += col_widths[j]
        y -= row_height

    # Bottom line
    ax.plot([x, x + total_w], [y, y], color=header_color, linewidth=0.5)
    return y


def draw_figure_page(pdf, img_path, title, caption=None):
    """Embed a PNG figure centered on a page."""
    fig, ax = new_page(pdf, title)
    img = mpimg.imread(img_path)
    h_img, w_img = img.shape[:2]
    aspect = w_img / h_img

    # Size to fit within margins
    avail_w = PAGE_W - 2 * MARGIN
    avail_h = PAGE_H - 2 * MARGIN - 0.8  # room for title + caption
    if avail_w / aspect <= avail_h:
        disp_w = avail_w
        disp_h = avail_w / aspect
    else:
        disp_h = avail_h
        disp_w = avail_h * aspect

    left = (PAGE_W - disp_w) / 2
    top = PAGE_H - MARGIN - 0.6
    img_ax = fig.add_axes([left / PAGE_W, (top - disp_h) / PAGE_H,
                           disp_w / PAGE_W, disp_h / PAGE_H])
    img_ax.imshow(img)
    img_ax.axis("off")

    if caption:
        draw_text_block(ax, MARGIN, top - disp_h - 0.25, caption,
                        fontsize=8, color="#555555", max_width=100)

    pdf.savefig(fig)
    plt.close(fig)


def build_pdf():
    pdf_path = os.path.join(ARTIFACTS_DIR, "propensity_analysis_report.pdf")

    # Load data
    t1_uw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "table1_unweighted.csv"))
    t1_w = pd.read_csv(os.path.join(ARTIFACTS_DIR, "table1_weighted.csv"))
    results = pd.read_csv(os.path.join(ARTIFACTS_DIR, "propensity_results.csv"))

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Title + Methods ──────────────────────────────────
        fig, ax = new_page(pdf)
        y = PAGE_H - MARGIN

        ax.text(PAGE_W / 2, y, "Propensity Analysis Report",
                ha="center", va="top", fontsize=18, fontweight="bold")
        y -= 0.35
        ax.text(PAGE_W / 2, y,
                "Heparin vs Enoxaparin Thromboprophylaxis after aSAH",
                ha="center", va="top", fontsize=13, color="#444444")
        y -= 0.25
        ax.text(PAGE_W / 2, y,
                "Pooled KSSG + MIMIC-IV Cohort  |  IPTW Analysis",
                ha="center", va="top", fontsize=10, color="#888888")
        y -= 0.6

        ax.plot([MARGIN, PAGE_W - MARGIN], [y, y], color="#cccccc", linewidth=0.5)
        y -= 0.35

        methods_text = (
            "Methods\n\n"
            "Study design: Inverse probability of treatment weighting (IPTW) was used to "
            "compare heparin vs enoxaparin at thromboprophylactic doses in a pooled cohort "
            "of 1,049 aSAH patients (875 heparin, 174 enoxaparin) from KSSG (n=179) and "
            "MIMIC-IV (n=870).\n\n"
            "Propensity score model: Logistic regression with 18 clinical covariates plus "
            "a site indicator (19 total). The model converged successfully with a "
            "c-statistic (AUC) of 0.995, reflecting the near-perfect prediction of "
            "treatment by site.\n\n"
            "IPTW weights: Stabilized weights were computed with propensity score trimming "
            "at [0.01, 0.99]. 834 scores were trimmed low and 52 trimmed high. Maximum "
            "weight: 17.41. Effective sample size: enoxaparin = 7.1, heparin = 458.3.\n\n"
            "Outcome analysis: IPTW-weighted generalized linear models (binomial family, "
            "logit link) with robust (HC1) standard errors. Results are reported as odds "
            "ratios (OR) with 95% confidence intervals. Three binary outcomes were "
            "analyzed: delayed cerebral ischemia (DCI), rebleeding, and in-hospital "
            "mortality.\n\n"
            "Sensitivity analysis: Treatment × site interaction terms were tested to "
            "evaluate whether treatment effects differed between sites."
        )

        y = draw_text_block(ax, MARGIN, y, "Methods", fontsize=13,
                            fontweight="bold", lineheight=0.25)
        y -= 0.15
        y = draw_text_block(ax, MARGIN, y, methods_text.split("\n\n", 1)[1],
                            fontsize=9, lineheight=0.17, max_width=95)

        # Key challenge box
        y -= 0.3
        from matplotlib.patches import FancyBboxPatch
        box_h = 1.3
        box = FancyBboxPatch((MARGIN, y - box_h), PAGE_W - 2 * MARGIN, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor="#fff3cd", edgecolor="#ffc107",
                              linewidth=1.5)
        ax.add_patch(box)
        y_box = y - 0.15
        draw_text_block(ax, MARGIN + 0.15, y_box,
                        "Key limitation: Site-treatment confounding",
                        fontsize=10, fontweight="bold", lineheight=0.2,
                        color="#856404")
        draw_text_block(ax, MARGIN + 0.15, y_box - 0.25,
                        "KSSG is 172/179 enoxaparin; MIMIC-IV is 868/870 heparin. "
                        "Site nearly perfectly predicts treatment assignment, producing "
                        "extreme propensity scores and very low effective sample size "
                        "(ESS = 7.1 for enoxaparin). IPTW cannot adequately balance "
                        "groups — 14 of 19 covariates retain SMD ≥ 0.1 after weighting. "
                        "All treatment effect estimates should be interpreted with "
                        "extreme caution.",
                        fontsize=8.5, lineheight=0.16, max_width=95,
                        color="#856404")

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Table 1 (Unweighted) ────────────────────────────
        fig, ax = new_page(pdf, "Table 1: Baseline Characteristics (Unweighted)")
        y = PAGE_H - MARGIN - 0.35

        col_widths = [2.2, 1.8, 1.8, 1.2]
        y = draw_table(ax, t1_uw, MARGIN, y, col_widths=col_widths)

        y -= 0.25
        draw_text_block(ax, MARGIN, y,
                        "Continuous variables: mean (SD). Binary variables: %. "
                        "SMD = absolute standardized mean difference.",
                        fontsize=8, color="#666666")

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Table 1 (Weighted) ──────────────────────────────
        fig, ax = new_page(pdf, "Table 1: Baseline Characteristics (IPTW-Weighted)")
        y = PAGE_H - MARGIN - 0.35

        y = draw_table(ax, t1_w, MARGIN, y, col_widths=col_widths)

        y -= 0.25
        draw_text_block(ax, MARGIN, y,
                        "Weighted means and proportions after stabilized IPTW. "
                        "Target: all SMD < 0.1. Covariates with SMD ≥ 0.1 indicate "
                        "residual imbalance.",
                        fontsize=8, color="#666666")

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Treatment Effects Table ──────────────────────────
        fig, ax = new_page(pdf, "Treatment Effects (IPTW-Adjusted)")
        y = PAGE_H - MARGIN - 0.35

        # Build display table
        effect_rows = []
        for _, r in results.iterrows():
            p_str = f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "<0.001"
            effect_rows.append({
                "Outcome": r["label"],
                "Enoxaparin": f"{r['rate_enoxaparin']:.1%}",
                "Heparin": f"{r['rate_heparin']:.1%}",
                "OR (95% CI)": f"{r['or']:.2f} ({r['or_lower']:.2f}\u2013{r['or_upper']:.2f})",
                "p": p_str,
            })
        effect_df = pd.DataFrame(effect_rows)
        col_widths_eff = [2.3, 1.2, 1.2, 1.8, 0.7]
        y = draw_table(ax, effect_df, MARGIN, y, col_widths=col_widths_eff,
                       fontsize=9, row_height=0.22)

        y -= 0.4
        y = draw_text_block(ax, MARGIN, y, "Interpretation", fontsize=11,
                            fontweight="bold", lineheight=0.25)
        y -= 0.12
        interp = (
            "DCI (OR 0.69, p=0.443): No significant difference between groups.\n\n"
            "Rebleeding (OR 2.82, p=0.134): Non-significant trend toward higher "
            "rebleeding odds with enoxaparin, but wide confidence interval "
            "(0.73\u201310.96) precludes meaningful inference.\n\n"
            "Mortality (OR 0.26, p=0.008): Statistically significant lower mortality "
            "odds with enoxaparin. However, given the ESS of only 7.1 in the "
            "enoxaparin group, 14/19 covariates with residual imbalance, and "
            "significant treatment \u00d7 site interaction (p<0.001), this result "
            "most likely reflects site-level differences rather than a true "
            "treatment effect."
        )
        y = draw_text_block(ax, MARGIN, y, interp, fontsize=9,
                            lineheight=0.17, max_width=95)

        # Sensitivity analysis
        y -= 0.4
        y = draw_text_block(ax, MARGIN, y, "Sensitivity: Treatment \u00d7 Site Interaction",
                            fontsize=11, fontweight="bold", lineheight=0.25)
        y -= 0.12
        sens_rows = [
            {"Outcome": "DCI", "Interaction coeff": "-47.606",
             "SE": "78.056", "p": "0.542"},
            {"Outcome": "Rebleeding", "Interaction coeff": "26.564",
             "SE": "1.395", "p": "<0.001"},
            {"Outcome": "Mortality", "Interaction coeff": "-19.766",
             "SE": "1.235", "p": "<0.001"},
        ]
        sens_df = pd.DataFrame(sens_rows)
        col_widths_sens = [2.0, 1.8, 1.0, 0.8]
        y = draw_table(ax, sens_df, MARGIN, y, col_widths=col_widths_sens,
                       fontsize=9, row_height=0.22)

        y -= 0.25
        draw_text_block(ax, MARGIN, y,
                        "Significant interaction for rebleeding and mortality "
                        "confirms that treatment effects differ by site, "
                        "undermining the validity of pooled estimates for "
                        "these outcomes.",
                        fontsize=8, color="#666666", max_width=95)

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: Love plot ────────────────────────────────────────
        draw_figure_page(
            pdf,
            os.path.join(ARTIFACTS_DIR, "love_plot.png"),
            "Covariate Balance: Before and After IPTW",
            caption="Figure 1. Absolute standardized mean differences (SMD) for each "
                    "covariate before (red) and after (blue) IPTW weighting. The dashed "
                    "line marks the conventional SMD = 0.1 threshold. Most covariates "
                    "retain SMD > 0.1 after weighting due to extreme site-treatment "
                    "confounding."
        )

        # ── Page 6: PS distribution ──────────────────────────────────
        draw_figure_page(
            pdf,
            os.path.join(ARTIFACTS_DIR, "ps_distribution.png"),
            "Propensity Score Distribution",
            caption="Figure 2. Distribution of propensity scores by treatment group. "
                    "Heparin patients cluster near PS = 0 and enoxaparin patients "
                    "near PS = 1, with minimal overlap. Dotted lines indicate the "
                    "trim thresholds [0.01, 0.99]."
        )

        # ── Page 7: Forest plot ──────────────────────────────────────
        draw_figure_page(
            pdf,
            os.path.join(ARTIFACTS_DIR, "forest_plot.png"),
            "IPTW-Adjusted Treatment Effects",
            caption="Figure 3. Forest plot of IPTW-adjusted odds ratios (enoxaparin "
                    "vs heparin) with 95% confidence intervals. The dashed line "
                    "indicates OR = 1 (no effect). Wide confidence intervals reflect "
                    "the low effective sample size."
        )

        # ── Page 8: Limitations ──────────────────────────────────────
        fig, ax = new_page(pdf, "Limitations")
        y = PAGE_H - MARGIN - 0.35

        limitations = [
            ("1. Site-treatment confounding",
             "KSSG patients predominantly received enoxaparin (172/179) while "
             "MIMIC-IV patients predominantly received heparin (868/870). Despite "
             "including site as a covariate, this near-perfect separation limits "
             "the propensity model\u2019s ability to fully balance groups. Treatment "
             "effects may reflect residual site-level confounding rather than "
             "true drug effects."),
            ("2. ICD-based outcome proxies",
             "DCI and rebleeding in MIMIC-IV are identified via ICD codes, which "
             "have lower sensitivity compared to clinician-verified diagnoses in "
             "KSSG. This may introduce differential outcome misclassification "
             "between sites."),
            ("3. Effective sample size reduction",
             "Stabilized IPTW with extreme propensity scores reduces the effective "
             "sample size to 7.1 for enoxaparin (from 174 actual patients) and "
             "458.3 for heparin (from 875). This dramatically widens confidence "
             "intervals and reduces statistical power."),
            ("4. Unmeasured confounders",
             "Institutional practice patterns, local protocols, ICU staffing, "
             "and temporal trends may differ between sites in ways not captured "
             "by measured covariates. The site indicator absorbs average "
             "differences but not heterogeneity in care quality."),
            ("5. Glucose units discrepancy",
             "KSSG glucose is in mmol/L while MIMIC-IV uses mg/dL, leading to a "
             "large pre-weighting SMD (3.16). While IPTW reduces this to 1.27, "
             "the residual imbalance reflects a measurement artifact rather than "
             "a true clinical difference."),
        ]

        for title_text, body_text in limitations:
            y = draw_text_block(ax, MARGIN, y, title_text,
                                fontsize=10, fontweight="bold", lineheight=0.2)
            y -= 0.05
            y = draw_text_block(ax, MARGIN + 0.15, y, body_text,
                                fontsize=9, lineheight=0.17, max_width=90)
            y -= 0.2

        # Conclusion box
        y -= 0.15
        box_h = 0.9
        box = FancyBboxPatch((MARGIN, y - box_h), PAGE_W - 2 * MARGIN, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor="#d4edda", edgecolor="#28a745",
                              linewidth=1.5)
        ax.add_patch(box)
        draw_text_block(ax, MARGIN + 0.15, y - 0.12,
                        "Conclusion",
                        fontsize=10, fontweight="bold", lineheight=0.2,
                        color="#155724")
        draw_text_block(ax, MARGIN + 0.15, y - 0.35,
                        "Due to the structural site-treatment confounding, this "
                        "IPTW analysis cannot reliably estimate the causal effect "
                        "of enoxaparin vs heparin. Results are hypothesis-generating "
                        "only and require validation in a setting where treatment "
                        "assignment is not determined by site.",
                        fontsize=9, lineheight=0.16, max_width=90,
                        color="#155724")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_pdf()
