"""Supplementary material: collects the aggregate outputs of the analysis scripts into one markdown document.

    results/<analysis>/*.csv, *.png, methods_*.md  ->  build_markdown(selection)  ->  pandoc  ->  PDF

Every item is tagged with the selections it belongs to: CORE (items the supplement will contain) or ALL
(every candidate). Numbering follows the order below within each selection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class Selection(Enum):
    ALL = 'all'
    CORE = 'core'


class Kind(Enum):
    METHODS = 'Methods'
    TABLE = 'Table'
    FIGURE = 'Figure'


BOTH = frozenset(Selection)
ALL_ONLY = frozenset({Selection.ALL})

CI_DASH = '–'
P_FLOOR = 0.001
MISSING = '—'
FIGURE_WIDTH = '90%'

COVARIATE_LABELS = {
    'age': 'Age (per year)',
    'male': 'Sex (male)',
    'hypertension': 'Hypertension',
    'poor_wfns': 'WFNS 4–5 (vs 1–3)',
    'fisher': 'Modified Fisher (per grade)',
    'active_smoker': 'Active smoking',
    'aspirin': 'Aspirin before SAH',
    'year': 'Calendar year (per year)',
    'alcohol': 'Alcohol abuse',
    'diabetes': 'Diabetes mellitus',
    'statin': 'Statin',
    'oral_anticoagulation': 'Oral anticoagulation',
}

# Sensitivity models of late DCI, as columns of one table: (file, model name in file or None, header, estimate)
LATE_DCI_SENSITIVITY_A = [
    ('primary_model.csv', 'primary', 'Primary (day 7)', 'HR'),
    ('landmark_sensitivity.csv', 'landmark day 5', 'Landmark day 5', 'HR'),
    ('landmark_sensitivity.csv', 'landmark day 10', 'Landmark day 10', 'HR'),
]
LATE_DCI_SENSITIVITY_B = [
    ('fine_gray.csv', None, 'Fine–Gray (sHR)', 'sHR'),
    ('sensitivity_cause_specific.csv', 'ICU discharge censoring', 'ICU discharge censoring', 'HR'),
    ('sensitivity_cause_specific.csv', 'no day-21 cap', 'No day-21 cap', 'HR'),
]

DCI_TIMING_OUTCOMES_METHODS = (
    'Sensitivity analyses of the association between DCI timing and outcome used the primary models (ordinal '
    'logistic regression of mRS 0–6; logistic regression of DCI-related infarction), each adjusted for age, sex, '
    'WFNS grade and modified Fisher grade, and changed one element at a time: (1) mRS from the 1-year visit or '
    'death only, without substitution by the 2- or 5-year visit; (2) survivors only, additionally adjusted for the '
    'interval from admission to the follow-up visit; (3) mRS at discharge; (4) additional adjustment for '
    'DCI-related infarction; (5) exclusion of in-hospital deaths; (6) additional adjustment for calendar year; '
    '(7) patients with a documented ictus date only. The association of DCI timing with missing follow-up mRS was '
    'assessed with logistic regression, and the association of DCI timing with admission severity with Spearman '
    'correlation and linear regression.'
)


@dataclass(frozen=True)
class Item:
    kind: Kind
    title: str
    legend: str
    render: Callable[[str], str]  # results_dir -> markdown body
    selections: frozenset


def _format_estimate(estimate: float, lower: float, upper: float) -> str:
    # e.g. 0.967, 0.948, 0.986 -> '0.97 (0.95–0.99)'
    if pd.isna(estimate):
        return MISSING
    return f'{estimate:.2f} ({lower:.2f}{CI_DASH}{upper:.2f})'


def _format_p(p: float) -> str:
    if pd.isna(p):
        return MISSING
    return f'<{P_FLOOR}' if p < P_FLOOR else f'{p:.3f}'


def _read(results_dir: str, *parts: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(results_dir, *parts))


def _markdown(table: pd.DataFrame) -> str:
    return table.to_markdown(index=False, disable_numparse=True)


def _estimate_columns(table: pd.DataFrame, estimate: str, label: str) -> pd.DataFrame:
    # Collapse estimate/lower/upper into 'HR (95% CI)' and format p
    formatted = _tidy(table.drop(columns=[estimate, 'lower', 'upper', 'p'], errors='ignore'), '.3f')
    formatted[f'{label} (95% CI)'] = [_format_estimate(*row) for row in table[[estimate, 'lower', 'upper']].values]
    if 'p' in table:
        formatted['p'] = table['p'].map(_format_p)
    if 'covariate' in formatted:
        formatted['covariate'] = formatted['covariate'].map(lambda name: COVARIATE_LABELS.get(name, name))
    return formatted


def _wide_models(models: list[tuple[str, str | None, str, str]], sub_dir: str) -> Callable[[str], str]:
    # One row per covariate, one column per model; e.g. 'Age (per year) | 0.97 (0.95–0.99) | ...'
    def render(results_dir: str) -> str:
        columns = {}
        for file, model, header, estimate in models:
            table = _read(results_dir, sub_dir, file)
            if model is not None:
                table = table[table['model'] == model]
            columns[header] = pd.Series([_format_estimate(*row) for row in table[[estimate, 'lower', 'upper']].values],
                                        index=table['covariate'].values)
        wide = pd.DataFrame(columns).fillna(MISSING)
        wide.index = [COVARIATE_LABELS.get(name, name) for name in wide.index]
        return _markdown(wide.rename_axis('Covariate').reset_index())
    return render


def _estimates(sub_dir: str, file: str, estimate: str, label: str, drop: tuple[str, ...] = ()) -> Callable[[str], str]:
    def render(results_dir: str) -> str:
        table = _read(results_dir, sub_dir, file).drop(columns=list(drop))
        return _markdown(_estimate_columns(table, estimate, label))
    return render


def _tidy(table: pd.DataFrame, float_format: str) -> pd.DataFrame:
    # Whole-number floats as integers (481.0 -> '481'), other floats rounded, NaN as a dash
    tidy = table.astype(object).copy()
    for column in table.select_dtypes('number'):
        values = table[column].dropna()
        whole = (values == values.round()).all()
        tidy[column] = [MISSING if pd.isna(value) else f'{value:.0f}' if whole else f'{value:{float_format}}'
                        for value in table[column]]
    return tidy.fillna(MISSING)


def _plain(sub_dir: str, file: str, float_format: str = '.3f') -> Callable[[str], str]:
    def render(results_dir: str) -> str:
        return _markdown(_tidy(_read(results_dir, sub_dir, file), float_format))
    return render


def _methods_file(file: str) -> Callable[[str], str]:
    # Drafted methods paragraph without its '# Methods: ...' heading
    def render(results_dir: str) -> str:
        with open(os.path.join(results_dir, file)) as handle:
            return ''.join(line for line in handle if not line.startswith('#')).strip()
    return render


def _figure(sub_dir: str, file: str) -> Callable[[str], str]:
    def render(results_dir: str) -> str:
        return f'![]({os.path.join(results_dir, sub_dir, file)}){{width={FIGURE_WIDTH}}}'
    return render


def _ridge(results_dir: str) -> str:
    table = _read(results_dir, 'late_dci', 'ridge_extended.csv').drop(columns=['model', 'bootstrap_fits'])
    return _markdown(_estimate_columns(table, 'HR', 'HR'))


def _model_check(results_dir: str) -> str:
    check = _read(results_dir, 'late_dci', 'model_check.csv').drop(columns=['bootstrap_failed'])
    groups = _read(results_dir, 'late_dci', 'calibration_groups.csv')
    return _markdown(_tidy(check, '.3f')) + '\n\n' + _markdown(_tidy(groups, '.3f'))


ITEMS = [
    # Methods
    Item(Kind.METHODS, 'Imaging use over calendar time', '', _methods_file('methods_imaging_over_time.md'), BOTH),
    Item(Kind.METHODS, 'DCI timing and outcome: sensitivity analyses', '',
         lambda _: DCI_TIMING_OUTCOMES_METHODS, BOTH),
    Item(Kind.METHODS, 'Factors associated with late-onset DCI (extended)', '',
         _methods_file('methods_late_onset_dci.md'), ALL_ONLY),

    # Tables: late-onset DCI
    Item(Kind.TABLE, 'Hypertension and aspirin: sequential adjustment',
         'Cause-specific Cox models of late DCI (day-7 landmark, complete case, n = 275, 66 events). Covariates are '
         'added stepwise; severity = WFNS grade and modified Fisher grade; full model = primary model.',
         _estimates('late_dci', 'sequential_adjustment.csv', 'HR', 'HR'), BOTH),
    Item(Kind.TABLE, 'Hypertension and aspirin: models with each exposure',
         'Primary model including hypertension only, aspirin only, or both.',
         _estimates('late_dci', 'exposure_specific.csv', 'HR', 'HR', drop=('n', 'events')), BOTH),
    Item(Kind.TABLE, 'Late DCI: alternative landmarks',
         'Cause-specific hazard ratios (95% CI) of late DCI from landmarks at day 5, 7 (primary) and 10.',
         _wide_models(LATE_DCI_SENSITIVITY_A, 'late_dci'), BOTH),
    Item(Kind.TABLE, 'Late DCI: alternative models and follow-up',
         'Fine–Gray subdistribution hazard ratios with death as competing event; cause-specific Cox with censoring at '
         'ICU discharge; cause-specific Cox without the day-21 cap.',
         _wide_models(LATE_DCI_SENSITIVITY_B, 'late_dci'), BOTH),
    Item(Kind.TABLE, 'Late DCI: ridge-penalised Cox model with the full covariate set of the initial analysis',
         'Penalty chosen by 5-fold cross-validation; 95% CI from 200 bootstrap samples. Clopidogrel excluded '
         '(fewer than 5 users).',
         _ridge, BOTH),
    Item(Kind.TABLE, 'Late DCI: effects before and after day 7',
         'Cox model from haemorrhage onset with effects split at day 7. Ratio = HR after / HR before day 7.',
         _plain('late_dci', 'piecewise.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'Late DCI: proportional hazards',
         'Schoenfeld residual test (rank time) for the primary model.',
         _plain('late_dci', 'proportional_hazards.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'Late DCI: hypertension and aspirin in the risk set',
         'Patients at risk on day 7 by hypertension and aspirin use; odds ratio and Fisher exact p for the '
         'association of hypertension with aspirin.',
         _plain('late_dci', 'htn_aspirin_association.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'Late DCI: calibration and discrimination at day 21',
         'Optimism corrected with 200 bootstrap samples; lower table: predicted and observed risk by quintile of '
         'predicted risk.',
         _model_check, ALL_ONLY),
    Item(Kind.TABLE, 'Late DCI: patient flow', 'Selection of the day-7 landmark risk set.',
         _plain('late_dci', 'flow.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'Late DCI: data sources and quality', 'Source of ictus date, death status and age.',
         _plain('late_dci', 'data_quality.csv'), ALL_ONLY),

    # Tables: DCI timing and outcome
    Item(Kind.TABLE, 'DCI timing and outcome: sensitivity analyses',
         'Odds ratio per day of later DCI onset; ordinal logistic regression of mRS 0–6 or logistic regression of '
         'DCI-related infarction, adjusted for age, sex, WFNS and modified Fisher grade unless stated. OR < 1: later '
         'DCI, lower mRS / less infarction.',
         _estimates('dci_timing_outcomes', 'sensitivity.csv', 'OR_per_day', 'OR', drop=('model', 'concern')), BOTH),
    Item(Kind.TABLE, 'DCI timing and outcome: primary and submitted models',
         '"submitted (probit)" reproduces the submitted manuscript (probit on mRS > 2, missing mRS counted as > 2). '
         '"logistic": ordinal on mRS 0–6 (infarction: binary), missing excluded. "logistic, mRS > 2": binary.',
         _estimates('dci_timing_outcomes', 'adjusted.csv', 'OR_per_day', 'OR'), ALL_ONLY),
    Item(Kind.TABLE, 'DCI timing and outcome: unadjusted comparison',
         'Median (IQR) days from haemorrhage to DCI; Mann–Whitney U.',
         _plain('dci_timing_outcomes', 'unadjusted.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'DCI timing and outcome: follow-up interval and mRS source',
         'Interval from admission to the follow-up visit used; deaths have no visit.',
         _plain('dci_timing_outcomes', 'follow_up_interval.csv'), ALL_ONLY),
    Item(Kind.TABLE, 'DCI timing and missing follow-up mRS',
         'Logistic regression of missing mRS on DCI day; adjusted model adds age, sex, WFNS and Fisher grade.',
         _estimates('dci_timing_outcomes', 'follow_up_missingness.csv', 'OR_per_day', 'OR'), ALL_ONLY),
    Item(Kind.TABLE, 'DCI timing and admission severity',
         'Spearman rho; linear regression of DCI day on age, sex, WFNS and Fisher grade (days per unit).',
         _plain('dci_timing_outcomes', 'severity.csv'), ALL_ONLY),

    # Tables: imaging over time
    Item(Kind.TABLE, 'Calendar-year trend in perfusion CT use',
         'Number of perfusion CTs: negative binomial, IRR per year; adjusted model adds DCI status with hospital days '
         'as exposure. At least one perfusion CT: logistic, OR per year.',
         _estimates('imaging_over_time', 'pct_trend.csv', 'per_year', 'per year'), ALL_ONLY),
    Item(Kind.TABLE, 'Calendar-year trend in perfusion parameter availability',
         'Patients with DCI; logistic regression, OR per year.',
         _estimates('imaging_over_time', 'perfusion_availability_trend.csv', 'per_year', 'OR per year',
                    drop=('model', 'estimate', 'warning')), ALL_ONLY),
    Item(Kind.TABLE, 'Calendar-year trend in verified DCI', 'Full cohort; logistic regression, OR per year.',
         _estimates('imaging_over_time', 'dci_trend.csv', 'per_year', 'OR per year', drop=('model', 'estimate')),
         ALL_ONLY),
    Item(Kind.TABLE, 'Perfusion CTs per patient by year', 'Mean with t-based 95% CI.',
         _plain('imaging_over_time', 'pct_by_year.csv', '.2f'), ALL_ONLY),

    # Figures
    Item(Kind.FIGURE, 'Imaging use over calendar time',
         '(A) Perfusion CTs per patient by year of haemorrhage. (B) Perfusion parameters reported at DCI diagnosis '
         'by year.',
         _figure('imaging_over_time', 'imaging_over_time.png'), BOTH),
    Item(Kind.FIGURE, 'Calibration of the late-DCI model at day 21',
         'Predicted versus observed cumulative incidence of late DCI by quintile of predicted risk, accounting for '
         'death as a competing risk.',
         _figure('late_dci', 'calibration.png'), BOTH),
    Item(Kind.FIGURE, 'Cumulative incidence of late DCI and death without DCI from day 7',
         'Aalen–Johansen estimates in the day-7 risk set (n = 310), overall and by WFNS grade.',
         _figure('late_dci', 'cumulative_incidence.png'), ALL_ONLY),
]


def build_markdown(results_dir: str, selection: Selection) -> str:
    """Numbered supplement for one selection; in ALL, items also in CORE are marked."""
    counters = {kind: 0 for kind in Kind}
    sections = {kind: [] for kind in Kind}
    for item in ITEMS:
        if selection not in item.selections:
            continue

        counters[item.kind] += 1
        marker = ' [core]' if selection == Selection.ALL and Selection.CORE in item.selections else ''
        heading = f'### Supplementary {item.kind.value} {counters[item.kind]}: {item.title}{marker}'
        body = item.render(results_dir)
        legend = f'{item.legend}\n\n' if item.legend else ''
        sections[item.kind].append(f'{heading}\n\n{legend}{body}\n')

    plural = {Kind.METHODS: 'Supplementary methods', Kind.TABLE: 'Supplementary tables',
              Kind.FIGURE: 'Supplementary figures'}
    return '\n'.join(f'## {plural[kind]}\n\n' + '\n'.join(sections[kind]) for kind in Kind if sections[kind])
