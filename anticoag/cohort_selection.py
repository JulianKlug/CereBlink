"""
Step 3: Cohort selection for aSAH anticoagulation study.

Applies sequential exclusion criteria, assigns treatment groups
(heparin vs enoxaparin), and generates a CONSORT-style flow chart.

Outputs:
  - cohort.csv             One row per included patient with group assignment
  - cohort_flow.csv        Exclusion counts at each step
  - cohort_flowchart.png   CONSORT flow diagram
"""

import sys
import warnings
from datetime import timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from utils import load_encrypted_xlsx, safe_conversion_to_datetime

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/mnt/data1/klug/datasets/kssg/SAH"
REGISTRY_PATH = f"{DATA_DIR}/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx"
OUTCOMES_PATH = f"{DATA_DIR}/outcomes_aSAH_DATA_2009_2024_18122024.xlsx"
CORRESPONDENCE_PATH = f"{DATA_DIR}/registry_pdms_correspondence.csv"

PDMS = f"{DATA_DIR}/extracted_data"
HEPARIN_DOSES_PATH = f"{PDMS}/20240207_SAH_SOS_EinzelGabeHeparinSpritzenpumpe.csv"
CLEXANE_DOSES_PATH = f"{PDMS}/20240207_SAH_SOS_EinzelGabeClexane.csv"
MEDIKAMENTE_PATH = f"{PDMS}/20250401_Medikamente.csv"

SECRETS_PATH = "/home/klug/icu_projects/CereBlink/.secrets"

# Thresholds
HEPARIN_THERAPEUTIC_UI_24H = 10_000
ENOXAPARIN_THERAPEUTIC_MG_DAY = 40


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_passwords():
    with open(SECRETS_PATH) as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_sources():
    """Load registry (with pNr linkage) and medication dose files."""
    passwords = load_passwords()

    # Registry
    registry = load_encrypted_xlsx(REGISTRY_PATH, password=passwords[1])

    # Correspondence
    corr = pd.read_csv(CORRESPONDENCE_PATH)
    corr.rename(columns={"JoinedName": "Name"}, inplace=True)
    corr["Date_birth"] = pd.to_datetime(
        corr["Date_birth"], format="%d.%m.%Y", errors="coerce"
    )

    # Merge registry <-> correspondence to get pNr
    registry["Date_birth"] = pd.to_datetime(registry["Date_birth"], errors="coerce")
    registry = registry.merge(
        corr[["SOS-CENTER-YEAR-NO.", "Name", "Date_birth", "pNr"]],
        on=["SOS-CENTER-YEAR-NO.", "Name", "Date_birth"],
        how="left",
    )

    # --- Medication files (combine dedicated files + general Medikamente) ---

    # Dedicated heparin file (Spritzenpumpe)
    hep_sp = pd.read_csv(HEPARIN_DOSES_PATH, sep=";", encoding="utf-8-sig")
    hep_sp = hep_sp[["pNr", "Start", "Ende", "Dauer", "Menge"]].copy()

    # General Medikamente file — heparin entries (excludes Heparin Creme)
    med = pd.read_csv(
        MEDIKAMENTE_PATH, sep=";", encoding="utf-8-sig", header=None,
        names=["pNr", "drugID", "drugName", "Start", "Ende", "Dauer",
               "Menge", "Menge2", "Einheit"],
    )
    hep_med = med[med["drugName"] == "Heparin"][["pNr", "Start", "Ende", "Dauer", "Menge"]].copy()

    # Combine and deduplicate heparin sources
    heparin = pd.concat([hep_sp, hep_med], ignore_index=True)
    heparin["pNr"] = pd.to_numeric(heparin["pNr"], errors="coerce")
    heparin.dropna(subset=["pNr"], inplace=True)
    heparin["pNr"] = heparin["pNr"].astype(int)
    heparin["Start"] = pd.to_datetime(heparin["Start"], errors="coerce")
    heparin["Ende"] = pd.to_datetime(heparin["Ende"], errors="coerce")
    heparin["Menge"] = pd.to_numeric(heparin["Menge"], errors="coerce")
    heparin["Dauer"] = pd.to_numeric(heparin["Dauer"], errors="coerce")
    heparin = heparin.drop_duplicates(subset=["pNr", "Start", "Ende", "Menge"])

    # Dedicated clexane file
    clex_file = pd.read_csv(CLEXANE_DOSES_PATH, sep=";", encoding="utf-8-sig")
    clex_file = clex_file[["pNr", "timeGabe", "Dosis"]].copy()

    # General Medikamente — clexane entries
    clex_med = med[med["drugName"] == "Clexane"][["pNr", "Start", "Menge"]].copy()
    clex_med.rename(columns={"Start": "timeGabe", "Menge": "Dosis"}, inplace=True)

    # Combine and deduplicate clexane sources
    clexane = pd.concat([clex_file, clex_med], ignore_index=True)
    clexane["pNr"] = pd.to_numeric(clexane["pNr"], errors="coerce")
    clexane.dropna(subset=["pNr"], inplace=True)
    clexane["pNr"] = clexane["pNr"].astype(int)
    clexane["timeGabe"] = pd.to_datetime(clexane["timeGabe"], errors="coerce")
    clexane["Dosis"] = pd.to_numeric(clexane["Dosis"], errors="coerce")
    clexane = clexane.drop_duplicates(subset=["pNr", "timeGabe", "Dosis"])

    return {"registry": registry, "heparin": heparin, "clexane": clexane}


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------
def find_duplicate_pnrs(registry):
    """
    Identify duplicate patients (same person, multiple registry entries).
    Returns set of pNr values to REMOVE (keep earliest admission per person).
    """
    df = registry.dropna(subset=["pNr"]).copy()
    df["pNr"] = df["pNr"].astype(int)
    df["Date_admission"] = pd.to_datetime(df["Date_admission"], errors="coerce")

    # Group by (Name, Date_birth) — same person
    grouped = df.groupby(["Name", "Date_birth"])
    remove_pnrs = set()
    for _, group in grouped:
        if len(group) > 1:
            # Keep earliest admission, remove the rest
            sorted_g = group.sort_values("Date_admission")
            remove_pnrs.update(sorted_g.iloc[1:]["pNr"].tolist())
    return remove_pnrs


def get_patients_with_drug(df):
    """Return set of pNr that have at least one dose record."""
    return set(df["pNr"].unique())


def check_enoxaparin_therapeutic(clexane_df):
    """Return set of pNr with any calendar day where total enoxaparin > 40 mg."""
    df = clexane_df.dropna(subset=["timeGabe", "Dosis"]).copy()
    df = df[df["Dosis"] > 0]
    df["date"] = df["timeGabe"].dt.date

    daily = df.groupby(["pNr", "date"])["Dosis"].sum().reset_index()
    therapeutic = daily[daily["Dosis"] > ENOXAPARIN_THERAPEUTIC_MG_DAY]
    return set(therapeutic["pNr"].unique())


def _split_infusion_to_daily(start, end, total_ie, duration_min):
    """
    Split a heparin infusion across calendar days proportionally.

    Returns list of (date, ie_amount) tuples.
    """
    if duration_min <= 0 or total_ie <= 0 or pd.isna(start) or pd.isna(end):
        return []
    if start >= end:
        return []

    rate_per_min = total_ie / duration_min
    result = []
    current = start

    while current < end:
        # Next midnight
        next_day = (current + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = min(end, next_day)
        minutes_in_day = (day_end - current).total_seconds() / 60
        ie_in_day = rate_per_min * minutes_in_day
        result.append((current.date(), ie_in_day))
        current = day_end

    return result


def check_heparin_therapeutic(heparin_df):
    """Return set of pNr with any calendar day where heparin > 10,000 UI."""
    df = heparin_df.dropna(subset=["Start", "Ende", "Menge", "Dauer"]).copy()
    df = df[(df["Menge"] > 0) & (df["Dauer"] > 0)]

    # Split each infusion across calendar days
    daily_totals = {}  # (pNr, date) -> total IE
    for _, row in df.iterrows():
        splits = _split_infusion_to_daily(
            row["Start"], row["Ende"], row["Menge"], row["Dauer"]
        )
        for date, ie in splits:
            key = (row["pNr"], date)
            daily_totals[key] = daily_totals.get(key, 0) + ie

    therapeutic = set()
    for (pnr, _), total_ie in daily_totals.items():
        if total_ie > HEPARIN_THERAPEUTIC_UI_24H:
            therapeutic.add(pnr)
    return therapeutic


def check_death_before_anticoag(registry, clexane_df, heparin_df):
    """Return set of pNr who died before receiving any anticoagulant."""
    reg = registry.dropna(subset=["pNr"]).copy()
    reg["pNr"] = reg["pNr"].astype(int)
    reg["Death"] = pd.to_numeric(reg["Death"], errors="coerce")
    reg["Date_admission"] = pd.to_datetime(reg["Date_admission"], errors="coerce")
    reg["Date_Death"] = pd.to_datetime(reg["Date_Death"], errors="coerce")
    reg["Days_to_Death"] = pd.to_numeric(reg["Days_to_Death"], errors="coerce")

    dead = reg[reg["Death"] == 1].set_index("pNr")

    # First anticoag date per patient
    first_clexane = (
        clexane_df.dropna(subset=["timeGabe"])
        .sort_values("timeGabe")
        .groupby("pNr")["timeGabe"]
        .first()
    )
    first_heparin = (
        heparin_df.dropna(subset=["Start"])
        .sort_values("Start")
        .groupby("pNr")["Start"]
        .first()
    )

    exclude = set()
    for pnr in dead.index:
        row = dead.loc[pnr] if isinstance(dead.loc[pnr], pd.Series) else dead.loc[pnr].iloc[0]

        # Determine death date
        death_date = row["Date_Death"]
        if pd.isna(death_date) and not pd.isna(row["Days_to_Death"]):
            death_date = row["Date_admission"] + timedelta(days=row["Days_to_Death"])

        # First anticoag date for this patient
        first_ac = None
        if pnr in first_clexane.index:
            first_ac = first_clexane[pnr]
        if pnr in first_heparin.index:
            h_date = first_heparin[pnr]
            if first_ac is None or h_date < first_ac:
                first_ac = h_date

        if first_ac is None:
            # Patient died and never received anticoag
            exclude.add(pnr)
        elif not pd.isna(death_date) and death_date <= first_ac:
            exclude.add(pnr)

    return exclude


# ---------------------------------------------------------------------------
# Main exclusion pipeline
# ---------------------------------------------------------------------------
def apply_exclusions(sources):
    """
    Apply sequential exclusion criteria.

    Returns (cohort_df, flow_steps).
    """
    registry = sources["registry"].copy()
    heparin = sources["heparin"]
    clexane = sources["clexane"]

    flow = []
    n_total = len(registry)
    flow.append({
        "step": 0,
        "description": "Total registry patients",
        "n_excluded": 0,
        "n_remaining": n_total,
    })
    print(f"  Step 0: Total registry patients: {n_total}")

    # --- Step 1: No PDMS linkage ---
    before = len(registry)
    registry = registry.dropna(subset=["pNr"]).copy()
    registry["pNr"] = registry["pNr"].astype(int)
    n_excl = before - len(registry)
    flow.append({
        "step": 1,
        "description": "No PDMS linkage",
        "n_excluded": n_excl,
        "n_remaining": len(registry),
    })
    print(f"  Step 1: No PDMS linkage — excluded {n_excl}, remaining {len(registry)}")

    # --- Step 2: Duplicates ---
    dup_pnrs = find_duplicate_pnrs(registry)
    before = len(registry)
    registry = registry[~registry["pNr"].isin(dup_pnrs)]
    n_excl = before - len(registry)
    flow.append({
        "step": 2,
        "description": "Duplicate patients",
        "n_excluded": n_excl,
        "n_remaining": len(registry),
    })
    print(f"  Step 2: Duplicates — excluded {n_excl}, remaining {len(registry)}")

    current_pnrs = set(registry["pNr"].unique())

    # --- Step 3: No anticoagulant received ---
    enox_pnrs = get_patients_with_drug(clexane)
    hep_pnrs = get_patients_with_drug(heparin)
    anticoag_pnrs = enox_pnrs | hep_pnrs
    no_anticoag = current_pnrs - anticoag_pnrs
    current_pnrs -= no_anticoag
    flow.append({
        "step": 3,
        "description": "No anticoagulant received",
        "n_excluded": len(no_anticoag),
        "n_remaining": len(current_pnrs),
    })
    print(f"  Step 3: No anticoagulant — excluded {len(no_anticoag)}, remaining {len(current_pnrs)}")

    # --- Step 4: Assign group by FIRST prophylactic agent ---
    # (patients who received both agents sequentially are classified by initial agent)
    first_clexane = (
        clexane.dropna(subset=["timeGabe"])
        .sort_values("timeGabe")
        .groupby("pNr")["timeGabe"]
        .first()
    )
    first_heparin = (
        heparin.dropna(subset=["Start"])
        .sort_values("Start")
        .groupby("pNr")["Start"]
        .first()
    )
    group_map = {}  # pNr -> group
    both_count = 0
    for pnr in current_pnrs:
        has_enox = pnr in first_clexane.index
        has_hep = pnr in first_heparin.index
        if has_enox and has_hep:
            both_count += 1
            # Classify by whichever came first
            if first_clexane[pnr] <= first_heparin[pnr]:
                group_map[pnr] = "enoxaparin"
            else:
                group_map[pnr] = "heparin"
        elif has_enox:
            group_map[pnr] = "enoxaparin"
        elif has_hep:
            group_map[pnr] = "heparin"
    n_hep_initial = sum(1 for g in group_map.values() if g == "heparin")
    n_enox_initial = sum(1 for g in group_map.values() if g == "enoxaparin")
    flow.append({
        "step": 4,
        "description": f"Group by initial agent ({both_count} received both sequentially)",
        "n_excluded": 0,
        "n_remaining": len(current_pnrs),
    })
    print(f"  Step 4: Group by initial agent — {both_count} received both (classified by first agent)")
    print(f"          Initial assignment: Heparin {n_hep_initial}, Enoxaparin {n_enox_initial}")

    # --- Step 5: Therapeutic anticoagulation ---
    # Check if initial agent was given at therapeutic dose
    therapeutic_enox = check_enoxaparin_therapeutic(
        clexane[clexane["pNr"].isin(current_pnrs)]
    )
    therapeutic_hep = check_heparin_therapeutic(
        heparin[heparin["pNr"].isin(current_pnrs)]
    )
    therapeutic = (therapeutic_enox | therapeutic_hep) & current_pnrs
    current_pnrs -= therapeutic
    # Also remove from group_map
    for pnr in therapeutic:
        group_map.pop(pnr, None)
    flow.append({
        "step": 5,
        "description": "Therapeutic anticoagulation",
        "n_excluded": len(therapeutic),
        "n_remaining": len(current_pnrs),
    })
    print(f"  Step 5: Therapeutic dose — excluded {len(therapeutic)}, remaining {len(current_pnrs)}")

    # --- Step 6: Death before anticoag start ---
    death_before = check_death_before_anticoag(
        registry[registry["pNr"].isin(current_pnrs)], clexane, heparin
    )
    death_before &= current_pnrs
    current_pnrs -= death_before
    flow.append({
        "step": 6,
        "description": "Death before anticoagulation",
        "n_excluded": len(death_before),
        "n_remaining": len(current_pnrs),
    })
    print(f"  Step 6: Death before anticoag — excluded {len(death_before)}, remaining {len(current_pnrs)}")

    # --- Step 7: Renal failure (placeholder) ---
    flow.append({
        "step": 7,
        "description": "Renal failure (creatinine >150) — data pending",
        "n_excluded": 0,
        "n_remaining": len(current_pnrs),
    })
    print(f"  Step 7: Renal failure — excluded 0 (data not yet available), remaining {len(current_pnrs)}")

    # --- Group assignment (from group_map built in step 4) ---
    cohort_rows = []
    reg_indexed = registry.set_index("pNr")
    for pnr in sorted(current_pnrs):
        group = group_map.get(pnr)
        if group is None:
            continue

        row_data = reg_indexed.loc[pnr]
        if isinstance(row_data, pd.DataFrame):
            row_data = row_data.iloc[0]

        cohort_rows.append({
            "pNr": pnr,
            "SOS-CENTER-YEAR-NO.": row_data["SOS-CENTER-YEAR-NO."],
            "group": group,
            "Date_admission": row_data["Date_admission"],
        })

    cohort = pd.DataFrame(cohort_rows)

    n_hep = (cohort["group"] == "heparin").sum()
    n_enox = (cohort["group"] == "enoxaparin").sum()
    flow.append({
        "step": "final",
        "description": f"Final cohort — Heparin: {n_hep}, Enoxaparin: {n_enox}",
        "n_excluded": 0,
        "n_remaining": len(cohort),
        "n_heparin": n_hep,
        "n_enoxaparin": n_enox,
    })

    return cohort, flow


# ---------------------------------------------------------------------------
# Flow chart
# ---------------------------------------------------------------------------
def generate_flowchart(flow_steps, output_path="cohort_flowchart.png"):
    """Generate a CONSORT-style flow chart."""
    # Filter to steps that actually exclude or are meaningful
    steps = [s for s in flow_steps if s["step"] != "final"]
    final = [s for s in flow_steps if s["step"] == "final"][0]

    # Only show exclusion steps that removed patients or are placeholders
    excl_steps = [s for s in steps[1:]
                  if s["n_excluded"] > 0 or "pending" in s["description"]]

    n_boxes = 2 + len(excl_steps)  # start + exclusions + final
    fig_h = max(10, 2.5 * n_boxes + 3)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(0, 10)
    y_max = 2.5 * n_boxes + 4
    ax.set_ylim(0, y_max)
    ax.axis("off")

    MAIN_COLOR = "#D6EAF8"
    EXCL_COLOR = "#FADBD8"
    FINAL_COLOR = "#D5F5E3"
    GROUP_COLOR = "#EBF5FB"
    BORDER = "#2C3E50"

    box_w = 3.5
    box_h = 0.8
    excl_w = 3.2
    excl_h = 0.6
    main_x = 4.0
    excl_x = 8.0
    y_step = 2.2

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=BORDER, linewidth=1.2,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                weight=weight)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2),
        )

    y = y_max - 1.5

    # Title
    ax.text(5, y + 1.0, "Cohort Selection Flow Chart",
            ha="center", fontsize=14, weight="bold")

    # Step 0: Starting population
    draw_box(main_x, y, box_w, box_h,
             f"Total registry patients\n(N = {steps[0]['n_remaining']})",
             MAIN_COLOR, fontsize=10, bold=True)

    # Exclusion steps
    for step in excl_steps:
        prev_y = y
        y -= y_step

        draw_arrow(main_x, prev_y - box_h / 2, main_x, y + box_h / 2)

        draw_box(main_x, y, box_w, box_h,
                 f"N = {step['n_remaining']}",
                 MAIN_COLOR, fontsize=10)

        mid_y = (prev_y + y) / 2
        if "pending" in step["description"]:
            excl_text = "Renal failure\n(data pending)"
        else:
            excl_text = f"{step['description']}\n(n = {step['n_excluded']})"

        draw_box(excl_x, mid_y, excl_w, excl_h, excl_text,
                 EXCL_COLOR, fontsize=8)
        draw_arrow(main_x + box_w / 2, mid_y,
                   excl_x - excl_w / 2, mid_y)

    # Final cohort
    prev_y = y
    y -= y_step
    draw_arrow(main_x, prev_y - box_h / 2, main_x, y + box_h / 2)
    draw_box(main_x, y, box_w, box_h,
             f"Final cohort\n(N = {final['n_remaining']})",
             FINAL_COLOR, fontsize=10, bold=True)

    # Group split
    group_y = y - y_step * 0.8
    hep_x = main_x - 2.0
    enox_x = main_x + 2.0

    draw_arrow(main_x - 0.3, y - box_h / 2, hep_x, group_y + box_h / 2)
    draw_arrow(main_x + 0.3, y - box_h / 2, enox_x, group_y + box_h / 2)

    draw_box(hep_x, group_y, 2.5, box_h,
             f"Heparin\n(n = {final.get('n_heparin', '?')})",
             GROUP_COLOR, fontsize=10, bold=True)
    draw_box(enox_x, group_y, 2.5, box_h,
             f"Enoxaparin\n(n = {final.get('n_enoxaparin', '?')})",
             GROUP_COLOR, fontsize=10, bold=True)

    # Note about group assignment
    ax.text(5, group_y - 0.7,
            "Groups assigned by first prophylactic agent received",
            ha="center", fontsize=8, style="italic", color="#666666")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFlow chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("COHORT SELECTION — aSAH Anticoagulation Study")
    print("=" * 80)

    print("\nLoading data sources...")
    sources = load_sources()
    print(f"  Registry: {len(sources['registry'])} patients")
    print(f"  Heparin dose records: {len(sources['heparin'])}")
    print(f"  Clexane dose records: {len(sources['clexane'])}")

    print("\nApplying exclusion criteria...")
    cohort, flow_steps = apply_exclusions(sources)

    n_hep = (cohort["group"] == "heparin").sum()
    n_enox = (cohort["group"] == "enoxaparin").sum()
    print(f"\n{'='*80}")
    print(f"FINAL COHORT: {len(cohort)} patients")
    print(f"  Heparin:    {n_hep}")
    print(f"  Enoxaparin: {n_enox}")
    print(f"{'='*80}")

    # Save outputs
    cohort.to_csv("cohort.csv", index=False)
    print(f"\nSaved cohort.csv ({len(cohort)} patients)")

    flow_df = pd.DataFrame(flow_steps)
    flow_df.to_csv("cohort_flow.csv", index=False)
    print(f"Saved cohort_flow.csv ({len(flow_df)} steps)")

    generate_flowchart(flow_steps)


if __name__ == "__main__":
    main()
