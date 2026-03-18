"""
Aggregate TIR & AUC burden analysis results into a single summary.

Walks all output directories under /mnt/data1/klug/output/cereblink_bp/tir_auc/,
parses regression result and fit CSVs, and produces:
  - tir_auc_summary.csv
  - tir_auc_summary.xlsx (multi-sheet)

FDR correction (Benjamini-Hochberg) is applied separately per
(outcome, analysis_type, time_period, bp_parameter).

Usage:
    python -m bp.bp_burden.summarize_tir_auc_results
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/mnt/data1/klug/output/cereblink_bp/tir_auc")

TIME_PERIODS = r"(?:before_aneurysm_secured|after_aneurysm_secured|before_dci|after_dci)"

# ── TIR filename patterns ────────────────────────────────────────────────
# Univariable per-bin: {bp}_tir_univariable_{outcome}_{period}_bin_{bin}_regression_{results|fit}.csv
TIR_UNI_RE = re.compile(
    r"^(?P<bp_param>systole|diastole|mitteldruck)"
    r"_tir_univariable"
    r"_(?P<outcome>DCI_YN_verified|mrs_1y)"
    r"_(?P<time_period>" + TIME_PERIODS + r")"
    r"_bin_(?P<bin>.+?)"
    r"_regression_(?P<filetype>results|fit)\.csv$"
)

# Multivariable: {bp}_tir_multivariable_{outcome}_{period}_regression_{results|fit}.csv
TIR_MULTI_RE = re.compile(
    r"^(?P<bp_param>systole|diastole|mitteldruck)"
    r"_tir_multivariable"
    r"_(?P<outcome>DCI_YN_verified|mrs_1y)"
    r"_(?P<time_period>" + TIME_PERIODS + r")"
    r"_regression_(?P<filetype>results|fit)\.csv$"
)

# ── AUC filename patterns ────────────────────────────────────────────────
# {bp}_auc_{above|below}_{outcome}_{period}_t{threshold}_{uni|multi}variable_regression_{results|fit}.csv
AUC_RE = re.compile(
    r"^(?P<bp_param>systole|diastole|mitteldruck)"
    r"_auc_(?P<direction>above|below)"
    r"_(?P<outcome>DCI_YN_verified|mrs_1y)"
    r"_(?P<time_period>" + TIME_PERIODS + r")"
    r"_t(?P<threshold>\d+)"
    r"_(?P<analysis_type>univariable|multivariable)"
    r"_regression_(?P<filetype>results|fit)\.csv$"
)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values for an array of p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]
    ranks = np.arange(1, n + 1)
    qvals = ranked_pvals * n / ranks
    for i in range(n - 2, -1, -1):
        qvals[i] = min(qvals[i], qvals[i + 1])
    qvals = np.minimum(qvals, 1.0)
    result = np.empty(n)
    result[order] = qvals
    return result


def parse_regression_results(path: Path) -> list[dict]:
    """Parse a _regression_results.csv, returning a list of variable rows."""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return []

    rows = []
    for var_name, row in df.iterrows():
        rows.append({
            "variable": str(var_name),
            "coef": row.get("coef"),
            "stderr": row.get("std err"),
            "z": row.get("z"),
            "pvalue": row.get("P>|z|"),
            "ci_lower": row.get("[0.025"),
            "ci_upper": row.get("0.975]"),
        })
    return rows


def parse_regression_fit(path: Path) -> dict:
    """Parse a _regression_fit.csv file."""
    info = {}
    try:
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                pairs = list(zip(row[0::2], row[1::2]))
                for key, val in pairs:
                    key = key.strip().rstrip(":")
                    val = val.strip()
                    if key:
                        info[key] = val
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    return {
        "model_type": info.get("Model", ""),
        "n_observations": safe_float(info.get("No. Observations")),
        "pseudo_r_squared": safe_float(info.get("Pseudo R-squ.")),
        "log_likelihood": safe_float(info.get("Log-Likelihood")),
        "aic": safe_float(info.get("AIC")),
        "bic": safe_float(info.get("BIC")),
    }


def _classify_subdir(subdir_name: str) -> dict:
    """Extract config flags from the subdirectory name."""
    name = subdir_name.lower()
    restrict_dci = "subgroup_dci" in name
    return {
        "run_config": subdir_name,
        "na_filtered": True,  # TIR/AUC pipeline always NA-filtered
        "subgroup": "dci_only" if restrict_dci else "all_patients",
    }


def collect_results() -> list[dict]:
    """Walk tir_auc output directories and collect all regression results."""
    summary_rows = []

    if not OUTPUT_ROOT.exists():
        logger.error("Output root %s does not exist", OUTPUT_ROOT)
        return summary_rows

    for subdir in sorted(OUTPUT_ROOT.iterdir()):
        if not subdir.is_dir():
            continue

        run_flags = _classify_subdir(subdir.name)
        results_files = list(subdir.rglob("*_regression_results.csv"))
        logger.info("Processing %s: %d results files", subdir.name, len(results_files))

        for results_path in results_files:
            fname = results_path.name
            fit_path = results_path.parent / fname.replace(
                "_regression_results.csv", "_regression_fit.csv"
            )

            meta = dict(run_flags)
            analysis_category = None  # 'tir' or 'auc'

            # Try TIR univariable
            m = TIR_UNI_RE.match(fname)
            if m:
                analysis_category = 'tir'
                meta.update({
                    "analysis_category": "tir",
                    "analysis_type": "univariable",
                    "bp_parameter": m.group("bp_param"),
                    "outcome": m.group("outcome"),
                    "time_period": m.group("time_period"),
                    "bin": m.group("bin"),
                    "direction": None,
                    "threshold": None,
                })

            # Try TIR multivariable
            if not analysis_category:
                m = TIR_MULTI_RE.match(fname)
                if m:
                    analysis_category = 'tir'
                    meta.update({
                        "analysis_category": "tir",
                        "analysis_type": "multivariable",
                        "bp_parameter": m.group("bp_param"),
                        "outcome": m.group("outcome"),
                        "time_period": m.group("time_period"),
                        "bin": "all",
                        "direction": None,
                        "threshold": None,
                    })

            # Try AUC
            if not analysis_category:
                m = AUC_RE.match(fname)
                if m:
                    analysis_category = 'auc'
                    meta.update({
                        "analysis_category": "auc",
                        "analysis_type": m.group("analysis_type"),
                        "bp_parameter": m.group("bp_param"),
                        "outcome": m.group("outcome"),
                        "time_period": m.group("time_period"),
                        "direction": m.group("direction"),
                        "threshold": int(m.group("threshold")),
                        "bin": None,
                    })

            if not analysis_category:
                logger.debug("Skipping unrecognized file: %s", fname)
                continue

            # Parse results
            var_rows = parse_regression_results(results_path)
            fit_info = parse_regression_fit(fit_path) if fit_path.exists() else {}

            # For TIR univariable, find the bin predictor row
            # For AUC, find the 'burden' predictor row
            # For multivariable, find the main predictor (first non-covariate, non-cutpoint)
            covariates_set = {'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping', 'const'}
            predictor_row = None
            for vr in var_rows:
                var = vr["variable"]
                # Skip cutpoints (ordinal model) and covariates
                if "/" in var or var in covariates_set:
                    continue
                predictor_row = vr
                break  # take the first non-covariate predictor

            if predictor_row is None:
                predictor_info = {
                    "predictor_variable": None,
                    "predictor_coef": None,
                    "predictor_stderr": None,
                    "predictor_z": None,
                    "predictor_pvalue": None,
                    "predictor_ci_lower": None,
                    "predictor_ci_upper": None,
                }
            else:
                predictor_info = {
                    "predictor_variable": predictor_row["variable"],
                    "predictor_coef": predictor_row["coef"],
                    "predictor_stderr": predictor_row["stderr"],
                    "predictor_z": predictor_row["z"],
                    "predictor_pvalue": predictor_row["pvalue"],
                    "predictor_ci_lower": predictor_row["ci_lower"],
                    "predictor_ci_upper": predictor_row["ci_upper"],
                }

            row = {**meta, **predictor_info, **fit_info}
            summary_rows.append(row)

    return summary_rows


def write_outputs(summary_rows: list[dict]) -> None:
    """Write CSV and multi-sheet Excel summary."""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        logger.error("No results collected — nothing to write.")
        return

    sort_cols = [c for c in [
        "run_config", "analysis_category", "outcome", "bp_parameter",
        "time_period", "analysis_type", "direction", "threshold", "bin",
    ] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ── FDR correction per (outcome, analysis_type, time_period, bp_parameter) ──
    df["fdr_qvalue"] = np.nan
    group_cols = ["outcome", "analysis_type", "time_period", "bp_parameter"]
    for _, idx in df.groupby(group_cols).groups.items():
        pvals = df.loc[idx, "predictor_pvalue"]
        valid = pvals.notna()
        if valid.sum() == 0:
            continue
        valid_idx = pvals[valid].index
        q = _benjamini_hochberg(pvals[valid].values.astype(float))
        df.loc[valid_idx, "fdr_qvalue"] = q
    df["fdr_significant"] = df["fdr_qvalue"] < 0.05

    # ── Write CSV ──
    csv_path = OUTPUT_ROOT / "tir_auc_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), csv_path)

    # ── Write Excel ──
    xlsx_path = OUTPUT_ROOT / "tir_auc_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)

        # Per analysis category
        for cat in ["tir", "auc"]:
            cat_df = df[df["analysis_category"] == cat]
            if not cat_df.empty:
                cat_df.to_excel(writer, sheet_name=f"{cat.upper()} Results", index=False)

        # Significant results (nominal)
        if "predictor_pvalue" in df.columns:
            sig_df = df[df["predictor_pvalue"] < 0.05]
            sig_df.to_excel(writer, sheet_name="Significant (p<0.05)", index=False)
            logger.info("  Nominally significant: %d / %d (%.0f%%)",
                        len(sig_df), len(df),
                        100 * len(sig_df) / len(df) if len(df) else 0)

        # FDR-corrected
        if "fdr_qvalue" in df.columns:
            fdr_df = df[df["fdr_significant"] == True]
            fdr_df.to_excel(writer, sheet_name="Significant (FDR<0.05)", index=False)
            logger.info("  FDR-significant: %d / %d (%.0f%%)",
                        len(fdr_df), len(df),
                        100 * len(fdr_df) / len(df) if len(df) else 0)

        # Per run config
        for config_name, group_df in df.groupby("run_config"):
            sheet_name = str(config_name)[:31]
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Wrote %s", xlsx_path)


def main():
    logger.info("Scanning %s ...", OUTPUT_ROOT)
    summary_rows = collect_results()
    write_outputs(summary_rows)

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        logger.info("--- Summary ---")
        logger.info("Total analyses: %d", len(df))
        logger.info("Analysis categories: %s", sorted(df.get("analysis_category", pd.Series()).unique()))
        logger.info("Run configs: %s", sorted(df["run_config"].unique()))
        logger.info("Outcomes: %s", sorted(df["outcome"].unique()))
        logger.info("BP parameters: %s", sorted(df["bp_parameter"].unique()))
        logger.info("Time periods: %s", sorted(df["time_period"].unique()))


if __name__ == "__main__":
    main()
