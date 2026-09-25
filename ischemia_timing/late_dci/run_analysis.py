"""Run the late-onset DCI analysis (revised analysis plan B).

From code/:
    python -m ischemia_timing.late_dci.run_analysis [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes aggregate tables (CSV), figures (PNG) and results.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from . import analyses
from .cohort import (CORE_COVARIATES, EXTENDED_COVARIATES, LANDMARK_DAY, AnalysisSet, FollowUpEnd,
                     build_landmark_dataset, build_patients, build_piecewise_dataset, select)
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'late_dci'))
SENSITIVITY_LANDMARKS = [5.0, 10.0]
FLOAT_FORMAT = '.3f'
FIGURE_DPI = 300


class Report:
    """Collects tables into CSV files and one markdown summary."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.sections: list[str] = []
        os.makedirs(output_dir, exist_ok=True)

    def table(self, name: str, title: str, table: pd.DataFrame, note: str = '') -> None:
        table.to_csv(os.path.join(self.output_dir, f'{name}.csv'), index=False)
        section = f'## {title}\n\n'
        if note:
            section += f'{note}\n\n'

        # Object dtype keeps counts as integers, e.g. n = 313 rather than 313.000
        printable = table.copy()
        for column in printable.select_dtypes(include=['integer', 'Int64']).columns:
            values = printable[column].astype(object)
            printable[column] = values.where(printable[column].notna(), '')
        section += printable.to_markdown(index=False, floatfmt=FLOAT_FORMAT) + '\n'
        self.sections.append(section)

    def text(self, title: str, body: str) -> None:
        self.sections.append(f'## {title}\n\n{body}\n')

    def write(self) -> None:
        with open(os.path.join(self.output_dir, 'results.md'), 'w') as file:
            file.write('# Late-onset DCI analysis (plan B)\n\n' + '\n'.join(self.sections))


def _data_quality(patients: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ('ictus from timings file', int((patients['ictus_source'] == 'timings').sum())),
        ('ictus recovered from outcomes file', int((patients['ictus_source'] == 'outcomes').sum())),
        ('ictus missing', int((patients['ictus_source'] == 'missing').sum())),
        ('death status from registry', int((patients['death_source'] == 'registry').sum())),
        ('death status recovered from outcomes file', int((patients['death_source'] == 'outcomes').sum())),
        ('death status missing', int((patients['death_source'] == 'missing').sum())),
        ('negative registry age recomputed', int(patients['age_recomputed'].sum())),
    ]
    return pd.DataFrame(rows, columns=['item', 'n'])


def _plot_cumulative_incidence(overall: pd.DataFrame, by_wfns: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for column, label in [('cif_event', 'late DCI'), ('cif_competing', 'death')]:
        axes[0].step(overall['time'] + LANDMARK_DAY, overall[column], where='post', label=label)
    axes[0].set_title('All patients at risk at day 7')

    for group, curve in by_wfns.groupby('group'):
        axes[1].step(curve['time'] + LANDMARK_DAY, curve['cif_event'], where='post', label=group.replace('poor_wfns=1', 'WFNS 4-5').replace('poor_wfns=0', 'WFNS 1-3'))
    axes[1].set_title('Late DCI by WFNS')

    for ax in axes:
        ax.set_xlabel('Days after ictus')
        ax.legend()
    axes[0].set_ylabel('Cumulative incidence')

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_calibration(calibration: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    limit = max(calibration['predicted'].max(), calibration['observed'].max()) * 1.1
    ax.plot([0, limit], [0, limit], linestyle='--', color='grey')
    ax.plot(calibration['predicted'], calibration['observed'], marker='o')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('Predicted risk of late DCI by day 21')
    ax.set_ylabel('Observed (Aalen-Johansen)')
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def _flow(selection_flow: pd.DataFrame, landmark_flow: pd.DataFrame) -> pd.Series:
    # landmark_flow starts with 'selected', already the last selection step
    steps = pd.concat([selection_flow, landmark_flow.iloc[1:]], ignore_index=True)
    return steps.set_index('step')['n']


def _flow_table(columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Side-by-side flows; step order from the longest flow, missing steps left blank."""
    order = max(columns.values(), key=len).index
    table = pd.DataFrame({label: flow.reindex(order) for label, flow in columns.items()}).astype('Int64')
    return table.rename_axis('step').reset_index()


def run(data_dir: str, secrets_path: str, output_dir: str) -> None:
    report = Report(output_dir)

    # Cohorts
    patients = build_patients(load_sources(data_dir, secrets_path))
    full = select(patients, AnalysisSet.FULL_COHORT)
    complete = select(patients, AnalysisSet.COMPLETE_CASE, CORE_COVARIATES)
    complete_extended = select(patients, AnalysisSet.COMPLETE_CASE, EXTENDED_COVARIATES)

    landmark_full = build_landmark_dataset(full.patients)
    landmark = build_landmark_dataset(complete.patients)
    risk_set = landmark.dataset

    report.table('data_quality', 'Data sources and quality', _data_quality(patients))
    flow = _flow_table({
        'full cohort': _flow(full.flow, landmark_full.flow),
        'complete case': _flow(complete.flow, landmark.flow),
    })
    report.table('flow', 'Patient flow (day-7 landmark, follow-up to discharge or day 21)', flow)

    # Cumulative incidence (full cohort)
    overall = analyses.cumulative_incidence(landmark_full.dataset)
    by_wfns = analyses.cumulative_incidence(landmark_full.dataset, 'poor_wfns')
    overall.to_csv(os.path.join(output_dir, 'cumulative_incidence.csv'), index=False)
    by_wfns.to_csv(os.path.join(output_dir, 'cumulative_incidence_by_wfns.csv'), index=False)
    _plot_cumulative_incidence(overall, by_wfns, os.path.join(output_dir, 'cumulative_incidence.png'))
    day21 = overall.iloc[-1]
    report.text('Cumulative incidence at day 21 (full cohort)',
                f'Late DCI {day21["cif_event"]:.3f}, competing death {day21["cif_competing"]:.3f}. '
                'Figure: `cumulative_incidence.png`.')

    # Primary model
    report.table('primary_model', 'Primary model: cause-specific Cox, complete case',
                 analyses.cause_specific_table(risk_set, CORE_COVARIATES, 'primary'))
    report.table('proportional_hazards', 'Proportional hazards (Schoenfeld, rank time)',
                 analyses.proportional_hazards_check(risk_set, CORE_COVARIATES))

    # Hypertension and aspirin
    report.table('sequential_adjustment', 'Hypertension and aspirin: sequential adjustment',
                 analyses.sequential_adjustment(risk_set))
    report.table('exposure_specific', 'Hypertension and aspirin: models with each exposure',
                 analyses.exposure_specific_models(risk_set))
    report.table('htn_aspirin_association', 'Hypertension-aspirin association in the risk set',
                 analyses.hypertension_aspirin_association(risk_set))

    # Early vs late
    report.table('piecewise', 'Piecewise Cox from ictus, split at day 7 (ratio = HR after / HR before)',
                 analyses.piecewise_contrast(build_piecewise_dataset(complete.patients)))
    landmark_tables = []
    for landmark_day in SENSITIVITY_LANDMARKS:
        data = build_landmark_dataset(complete.patients, landmark_day=landmark_day).dataset
        landmark_tables.append(analyses.cause_specific_table(data, CORE_COVARIATES, f'landmark day {landmark_day:g}'))
    report.table('landmark_sensitivity', 'Landmarks at day 5 and day 10', pd.concat(landmark_tables, ignore_index=True))

    # Sensitivity analyses
    report.table('fine_gray', 'Sensitivity 1: Fine-Gray (subdistribution HR)', analyses.fine_gray_table(risk_set, CORE_COVARIATES))

    ridge = analyses.ridge_model(build_landmark_dataset(complete_extended.patients).dataset, EXTENDED_COVARIATES)
    note = 'CI: bootstrap percentiles at the CV-selected penalty.'
    if ridge.dropped_sparse:
        note += (f' Dropped (< {analyses.MIN_PATIENTS_PER_LEVEL} patients in a level): '
                 f'{", ".join(ridge.dropped_sparse)}.')
    report.table('ridge_extended', 'Sensitivity 2: ridge Cox with the original full covariate set', ridge.table, note)

    variants = [
        ('ICU discharge censoring', build_landmark_dataset(complete.patients, follow_up_end=FollowUpEnd.ICU_DISCHARGE)),
        ('no day-21 cap', build_landmark_dataset(complete.patients, cap_day=None)),
    ]
    tables = [analyses.cause_specific_table(v.dataset, CORE_COVARIATES, label) for label, v in variants]
    report.table('sensitivity_cause_specific', 'Sensitivity 3-4: cause-specific Cox variants', pd.concat(tables, ignore_index=True))

    # Model check
    check = analyses.model_check(risk_set, CORE_COVARIATES)
    report.table('model_check', 'Model check at day 21 (bootstrap optimism correction)', check.metrics,
                 f'Death nuisance model: ridge penalizer {analyses.DEATH_MODEL_PENALIZER}.')
    report.table('calibration_groups', 'Calibration by predicted-risk quintile', check.calibration)
    _plot_calibration(check.calibration, os.path.join(output_dir, 'calibration.png'))

    report.write()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=FutureWarning)
    run(args.data_dir, args.secrets, args.output_dir)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
