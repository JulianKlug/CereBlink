"""
Aggregate BP burden analysis results from all pipeline runs into a single summary.

Walks all output directories under /mnt/data1/klug/output/cereblink_bp/,
parses regression result and fit CSVs, and produces:
  - bp_burden_summary.csv  (flat CSV, one row per analysis)
  - bp_burden_summary.xlsx (Excel with multiple sheets)

Usage:
    python -m bp.bp_burden.summarize_results          # from CereBlink root
    python bp/bp_burden/summarize_results.py           # also works standalone
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/mnt/data1/klug/output/cereblink_bp")
EXCLUDE_DIRS = {"old_output", "test", "test_run"}

# ── Filename regex ──────────────────────────────────────────────────────────
# Example: "diastole_in hypotension episodes_DCI_YN_verified_t0_after_24h_
#           decision_boundary_analysis_neg_event_duration_multivariable_
#           regression_results.csv"
TIME_PERIODS = (
    "first_24h|after_24h|before_aneurysm_secured|after_aneurysm_secured"
    "|before_dci|after_dci"
)
FILENAME_RE = re.compile(
    r"^(?P<bp_param>systole|diastole|mitteldruck)"
    r"_in (?P<direction>hypertension|hypotension) episodes"
    r"_(?P<outcome>DCI_YN_verified|mrs_1y)"
    r"_t0_(?P<time_period>" + TIME_PERIODS + r")"
    r"_decision_boundary_analysis"
    r"_(?P<event_corr>pos|neg)_event_duration"
    r"(?:_(?P<multivariable>multivariable))?"
    r"_regression_(?P<filetype>results|fit)\.csv$"
)


def classify_run_config(dirname: str) -> dict:
    """Derive na_filtered and subgroup flags from a top-level directory name."""
    # Normalize: replace underscores with spaces for matching
    name_lower = dirname.lower().replace("_", " ")
    na_filtered = "na filter" in name_lower
    if "subgroup non dci" in name_lower or "restricted to non dci" in name_lower or "restrictet to non dci" in name_lower or "subgorup restricted to non dci" in name_lower:
        subgroup = "non_dci"
    elif "subgroup dci" in name_lower or "restricted to dci" in name_lower or "restrictet to dci" in name_lower:
        subgroup = "dci_only"
    else:
        subgroup = "all_patients"
    return {"na_filtered": na_filtered, "subgroup": subgroup}


def parse_regression_results(path: Path) -> tuple[dict | None, list[dict]]:
    """Parse a _regression_results.csv file.

    Returns (predictor_row, covariate_rows) where predictor_row contains the
    main predictor (the *_correlated_event_proportion_of_monitoring_duration
    variable) and covariate_rows has everything else (covariates + cutpoints).
    """
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None, []

    predictor_row = None
    covariate_rows = []
    for var_name, row in df.iterrows():
        var_name = str(var_name)
        if "correlated_event_proportion_of_monitoring_duration" in var_name:
            predictor_row = {
                "predictor_coef": row.get("coef"),
                "predictor_stderr": row.get("std err"),
                "predictor_z": row.get("z"),
                "predictor_pvalue": row.get("P>|z|"),
                "predictor_ci_lower": row.get("[0.025"),
                "predictor_ci_upper": row.get("0.975]"),
            }
        else:
            covariate_rows.append(
                {
                    "variable": var_name,
                    "coef": row.get("coef"),
                    "stderr": row.get("std err"),
                    "z": row.get("z"),
                    "pvalue": row.get("P>|z|"),
                    "ci_lower": row.get("[0.025"),
                    "ci_upper": row.get("0.975]"),
                }
            )
    return predictor_row, covariate_rows


def parse_regression_fit(path: Path) -> dict:
    """Parse a _regression_fit.csv file.

    Handles both Logit (statsmodels summary2) and OrderedModel layouts.
    These CSVs have a grid layout with key-value pairs across columns.
    """
    info = {}
    try:
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                # Each row can have up to 4 cells: key1, val1, key2, val2
                pairs = list(zip(row[0::2], row[1::2]))
                for key, val in pairs:
                    key = key.strip().rstrip(":")
                    val = val.strip()
                    if not key:
                        continue
                    info[key] = val
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    result = {
        "model_type": info.get("Model", ""),
        "n_observations": safe_float(info.get("No. Observations")),
        "pseudo_r_squared": safe_float(info.get("Pseudo R-squ.")),
        "log_likelihood": safe_float(info.get("Log-Likelihood")),
        "aic": safe_float(info.get("AIC")),
        "bic": safe_float(info.get("BIC")),
        "converged": info.get("converged"),
        "llr_pvalue": safe_float(info.get("LLR p-value")),
    }
    return result


def collect_results() -> tuple[list[dict], list[dict]]:
    """Walk all output directories and collect results."""
    summary_rows = []
    covariate_rows = []

    for top_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not top_dir.is_dir() or top_dir.name in EXCLUDE_DIRS:
            continue

        run_config_name = top_dir.name
        run_flags = classify_run_config(run_config_name)

        # Find all regression_results CSVs under this top-level dir
        results_files = list(top_dir.rglob("*_regression_results.csv"))
        logger.info(
            "Processing %s: %d results files", run_config_name, len(results_files)
        )

        for results_path in results_files:
            m = FILENAME_RE.match(results_path.name)
            if not m:
                logger.warning("Unparseable filename: %s", results_path.name)
                continue

            # Build the corresponding fit file path
            fit_filename = results_path.name.replace(
                "_regression_results.csv", "_regression_fit.csv"
            )
            fit_path = results_path.parent / fit_filename

            # Extract metadata from filename
            analysis_type = (
                "multivariable" if m.group("multivariable") else "univariable"
            )
            meta = {
                "run_config": run_config_name,
                "outcome": m.group("outcome"),
                "na_filtered": run_flags["na_filtered"],
                "subgroup": run_flags["subgroup"],
                "bp_parameter": m.group("bp_param"),
                "direction": m.group("direction"),
                "time_period": m.group("time_period"),
                "event_correlation": m.group("event_corr"),
                "analysis_type": analysis_type,
            }

            # Parse results
            predictor, covariates = parse_regression_results(results_path)
            if predictor is None:
                logger.warning("No predictor row found in %s", results_path)
                predictor = {
                    k: None
                    for k in [
                        "predictor_coef",
                        "predictor_stderr",
                        "predictor_z",
                        "predictor_pvalue",
                        "predictor_ci_lower",
                        "predictor_ci_upper",
                    ]
                }

            # Parse fit
            fit_info = parse_regression_fit(fit_path) if fit_path.exists() else {}

            row = {**meta, **predictor, **fit_info}
            summary_rows.append(row)

            # Collect covariate data for multivariable models
            if analysis_type == "multivariable" and covariates:
                for cov in covariates:
                    covariate_rows.append({**meta, **cov})

    return summary_rows, covariate_rows


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values for an array of p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]
    # q_i = p_i * n / rank_i, then enforce monotonicity from the bottom up
    ranks = np.arange(1, n + 1)
    qvals = ranked_pvals * n / ranks
    # Enforce monotonicity: q[i] = min(q[i], q[i+1])
    for i in range(n - 2, -1, -1):
        qvals[i] = min(qvals[i], qvals[i + 1])
    qvals = np.minimum(qvals, 1.0)
    # Restore original order
    result = np.empty(n)
    result[order] = qvals
    return result


def write_outputs(
    summary_rows: list[dict], covariate_rows: list[dict]
) -> None:
    """Write CSV and Excel summary files."""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        logger.error("No results collected — nothing to write.")
        return

    # Sort for readability
    sort_cols = [
        "run_config",
        "outcome",
        "bp_parameter",
        "direction",
        "time_period",
        "event_correlation",
        "analysis_type",
    ]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Benjamini-Hochberg FDR correction, applied separately per (outcome, analysis_type)
    df["fdr_qvalue"] = np.nan
    for _, idx in df.groupby(["outcome", "analysis_type", "na_filtered", "time_period"]).groups.items():
        pvals = df.loc[idx, "predictor_pvalue"]
        valid = pvals.notna()
        if valid.sum() == 0:
            continue
        valid_idx = pvals[valid].index
        q = _benjamini_hochberg(pvals[valid].values)
        df.loc[valid_idx, "fdr_qvalue"] = q
    df["fdr_significant"] = df["fdr_qvalue"] < 0.05

    # Flag rows where heatmap-derived event correlation direction disagrees with
    # the regression predictor coefficient sign (expected due to Simpson's paradox)
    df["opposite_direction"] = (
        ((df["event_correlation"] == "pos") & (df["predictor_coef"] < 0))
        | ((df["event_correlation"] == "neg") & (df["predictor_coef"] > 0))
    )

    # Write CSV
    csv_path = OUTPUT_ROOT / "bp_burden_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), csv_path)

    # Write Excel with multiple sheets
    xlsx_path = OUTPUT_ROOT / "bp_burden_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)

        # One sheet per run config
        for config_name, group_df in df.groupby("run_config"):
            # Excel sheet names max 31 chars
            sheet_name = str(config_name)[:31]
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Significant results (nominal)
        if "predictor_pvalue" in df.columns:
            sig_df = df[df["predictor_pvalue"] < 0.05]
            sig_df.to_excel(writer, sheet_name="Significant (p<0.05)", index=False)
            logger.info(
                "  Nominally significant: %d / %d (%.0f%%)",
                len(sig_df),
                len(df),
                100 * len(sig_df) / len(df) if len(df) else 0,
            )

        # Significant results (FDR-corrected)
        if "fdr_qvalue" in df.columns:
            fdr_df = df[df["fdr_significant"] == True]
            fdr_df.to_excel(writer, sheet_name="Significant (FDR<0.05)", index=False)
            logger.info(
                "  FDR-significant: %d / %d (%.0f%%)",
                len(fdr_df),
                len(df),
                100 * len(fdr_df) / len(df) if len(df) else 0,
            )

        # Covariates sheet
        if covariate_rows:
            cov_df = pd.DataFrame(covariate_rows)
            cov_df = cov_df.sort_values(
                sort_cols + ["variable"]
            ).reset_index(drop=True)
            cov_df.to_excel(writer, sheet_name="Covariates", index=False)
            logger.info("  Covariate rows: %d", len(cov_df))

    logger.info("Wrote %s", xlsx_path)


def main():
    logger.info("Scanning %s ...", OUTPUT_ROOT)
    summary_rows, covariate_rows = collect_results()
    write_outputs(summary_rows, covariate_rows)

    # Quick summary stats
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        logger.info("--- Summary ---")
        logger.info("Total analyses: %d", len(df))
        logger.info("Run configs: %s", sorted(df["run_config"].unique()))
        logger.info("Outcomes: %s", sorted(df["outcome"].unique()))
        logger.info("BP parameters: %s", sorted(df["bp_parameter"].unique()))
        logger.info("Time periods: %s", sorted(df["time_period"].unique()))


if __name__ == "__main__":
    main()
