"""Circumstances of DCI diagnosis: assessability, pCT verification and triggers, and ascertainment analyses.

From code/:
    python -m ischemia_timing.late_dci.run_dci_triggers [--late_case_list PATH] [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes aggregate tables (CSV) and results.md; no patient-level data. --late_case_list writes the IDs and onset of
DCI after day 14 to PATH, for chart verification only; keep it outside results/.
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

from . import dci_triggers
from .cohort import ID, AnalysisSet, build_patients, select
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'dci_triggers'))
FLOAT_FORMAT = '.2f'
LATE_CASE_COLUMNS = [ID, 'Date_Ictus', 'Date_DCI_ischemia_first_image']

NOTE = ('Verified DCI patients, full cohort. Percent of recorded values with Wilson 95% CI. Monitoring triggers '
        '(ICP, PtiO2, microdialysis, NIRS): blank = not monitored, counted as no (n_blank_as_no); TCD (every patient) and other '
        'items: blank = missing. pCT triggers are not mutually exclusive.')
PLAN = 'Plan: late_dci_analysis/analysis_plan_dci_ascertainment.md. Exploratory; no multiplicity correction.'


def _markdown(title: str, table: pd.DataFrame, note: str) -> str:
    return f'## {title}\n\n{note}\n\n' + table.to_markdown(index=False, floatfmt=FLOAT_FORMAT) + '\n'


def _write_late_cases(sources, patients: pd.DataFrame, path: str) -> None:
    # Row labels of the patient table are row labels of the timings file
    cases = sources.dci_timings.loc[dci_triggers.late_cases(patients), LATE_CASE_COLUMNS]
    cases.assign(onset_day=patients.loc[cases.index, 't_dci']).sort_values('onset_day').to_csv(path, index=False)


def run(data_dir: str, secrets_path: str, output_dir: str, late_case_list: str | None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sources = load_sources(data_dir, secrets_path)
    patients = select(build_patients(sources), AnalysisSet.FULL_COHORT).patients

    onset = dci_triggers.onset_by_source(patients)
    by_year = dci_triggers.source_by_year(patients)
    tail = dci_triggers.late_tail(patients)
    tables = [
        ('diagnosis_circumstances', 'Circumstances of DCI diagnosis', dci_triggers.diagnosis_circumstances(patients), NOTE),
        ('device_use', '1. Devices in use at DCI diagnosis', dci_triggers.device_use(patients),
         'All verified DCI. In use = cell filled (TCD: every patient). Positive among recorded. Wilson 95% CI; Fisher, not assessable vs assessable.'),
        ('onset_groups', '2. Onset day by trigger source and assessability', onset.groups, 'Days from ictus, median (IQR).'),
        ('onset_comparisons', '2. Onset difference, exposed - reference (days)', onset.comparisons,
         'Hodges-Lehmann difference, distribution-free 95% CI; Mann-Whitney U. Adjusted: median regression + poor WFNS, '
         f'age, calendar year; percentile bootstrap 95% CI ({dci_triggers.BOOTSTRAP_SAMPLES} samples).'),
        ('source_by_period', '3. Ascertainment by period', by_year.by_period, 'n yes / n, Wilson 95% CI.'),
        ('source_trend', '3. Ascertainment by calendar year', by_year.trend,
         'Logistic, OR per year. PtiO2, microdialysis and NIRS descriptive only (too few in use).'),
        ('year_on_onset', '3. Calendar year and onset day', by_year.year_on_onset,
         'pCT verified, any monitoring or clinical only, complete covariates. Median regression + poor WFNS, age; '
         'days per year, bootstrap 95% CI.'),
        ('late_tail_comparison', f'4. Onset after day {dci_triggers.LATE_TAIL_DAY:g} vs earlier', tail.comparison,
         'n yes / n; Fisher exact.'),
        ('late_tail_distribution', '4. Onset distribution by ascertainment', tail.distribution,
         'Clinically detected, assessable: assessable and clinical trigger only.'),
    ]

    sections = []
    for name, title, table, note in tables:
        table.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
        sections.append(_markdown(title, table, note))

    with open(os.path.join(output_dir, 'results.md'), 'w') as file:
        file.write(f'# Circumstances of DCI diagnosis\n\n{PLAN}\n\n' + '\n'.join(sections))

    if late_case_list:
        _write_late_cases(sources, patients, late_case_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--late_case_list', default=None, help='patient-level CSV of DCI after day 14 (internal)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=FutureWarning)
    run(args.data_dir, args.secrets, args.output_dir, args.late_case_list)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
