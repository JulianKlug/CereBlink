"""Analyses of revised analysis plan B. Each function returns tidy DataFrames."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import k_fold_cross_validation
from scipy import stats

from . import competing_risks as cr
from .cohort import CORE_COVARIATES, FOLLOW_UP_CAP_DAY, LANDMARK_DAY, Event

RANDOM_SEED = 2026
BOOTSTRAP_SAMPLES = 200
CV_FOLDS = 5
RIDGE_PENALIZERS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
CALIBRATION_GROUPS = 5
MIN_PATIENTS_PER_LEVEL = 5
CI_PERCENTILES = (2.5, 97.5)
Z_95 = stats.norm.ppf(0.975)

# Death model is only a nuisance model for absolute risk; 17 deaths / 8 df -> mild ridge
DEATH_MODEL_PENALIZER = 0.1

# Model check horizon: day 21 = 14 days after the day-7 landmark
HORIZON = FOLLOW_UP_CAP_DAY - LANDMARK_DAY

PIECEWISE_COVARIATES = ['hypertension', 'age', 'poor_wfns']
EXPOSURES = ['hypertension', 'aspirin']
SEQUENTIAL_ADJUSTMENT = [
    ('crude', []),
    ('+ age', ['age']),
    ('+ severity', ['age', 'poor_wfns', 'fisher']),
    ('+ calendar year', ['age', 'poor_wfns', 'fisher', 'year']),
    ('full model', None),  # all core covariates
]


def _event_indicator(dataset: pd.DataFrame, cause: Event) -> pd.DataFrame:
    return dataset.assign(is_event=(dataset['event'] == cause).astype(int))


def fit_cause_specific(dataset: pd.DataFrame, covariates: list[str], cause: Event = Event.DCI,
                       penalizer: float = 0.0) -> CoxPHFitter:
    data = _event_indicator(dataset, cause)[['time', 'is_event'] + covariates]
    fitter = CoxPHFitter(penalizer=penalizer)
    fitter.fit(data, duration_col='time', event_col='is_event')
    return fitter


def hazard_ratios(summary: pd.DataFrame, model: str) -> pd.DataFrame:
    table = summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    table.columns = ['HR', 'lower', 'upper', 'p']
    return table.rename_axis('covariate').reset_index().assign(model=model)[['model', 'covariate', 'HR', 'lower', 'upper', 'p']]


def cause_specific_table(dataset: pd.DataFrame, covariates: list[str], model: str) -> pd.DataFrame:
    fitter = fit_cause_specific(dataset, covariates)
    events = int((dataset['event'] == Event.DCI).sum())
    return hazard_ratios(fitter.summary, model).assign(n=len(dataset), events=events)


def proportional_hazards_check(dataset: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Schoenfeld-residual test against rank-transformed time."""
    fitter = fit_cause_specific(dataset, covariates)
    data = _event_indicator(dataset, Event.DCI)[['time', 'is_event'] + covariates]
    result = proportional_hazard_test(fitter, data, time_transform='rank')
    return result.summary[['test_statistic', 'p']].rename_axis('covariate').reset_index()


def sequential_adjustment(dataset: pd.DataFrame) -> pd.DataFrame:
    """HR of each exposure as adjustment is added step by step."""
    rows = []
    for exposure in EXPOSURES:
        for step, adjustment in SEQUENTIAL_ADJUSTMENT:
            covariates = CORE_COVARIATES if adjustment is None else [exposure] + adjustment
            table = cause_specific_table(dataset, covariates, step)
            rows.append(table[table['covariate'] == exposure].assign(exposure=exposure))

    return pd.concat(rows, ignore_index=True)[['exposure', 'model', 'HR', 'lower', 'upper', 'p']]


def exposure_specific_models(dataset: pd.DataFrame) -> pd.DataFrame:
    """Full model with hypertension only, aspirin only, and both."""
    variants = [
        ('hypertension only', [c for c in CORE_COVARIATES if c != 'aspirin']),
        ('aspirin only', [c for c in CORE_COVARIATES if c != 'hypertension']),
        ('both', CORE_COVARIATES),
    ]
    rows = [cause_specific_table(dataset, covariates, label) for label, covariates in variants]
    table = pd.concat(rows, ignore_index=True)
    return table[table['covariate'].isin(EXPOSURES)].reset_index(drop=True)


def hypertension_aspirin_association(dataset: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(dataset['hypertension'], dataset['aspirin'])
    odds_ratio, p = stats.fisher_exact(counts.to_numpy())

    late_dci = dataset['event'] == Event.DCI
    rows = []
    for (htn, asa), group in dataset.groupby(['hypertension', 'aspirin']):
        rows.append((int(htn), int(asa), len(group), int(late_dci[group.index].sum())))

    table = pd.DataFrame(rows, columns=['hypertension', 'aspirin', 'n', 'late_dci'])
    return table.assign(odds_ratio_htn_aspirin=odds_ratio, fisher_p=p)


def piecewise_contrast(piecewise_rows: pd.DataFrame) -> pd.DataFrame:
    """HR before vs after the split for selected covariates; ratio of HRs with 95% CI."""
    data = piecewise_rows.copy()
    covariates = []
    for covariate in CORE_COVARIATES:
        if covariate not in PIECEWISE_COVARIATES:
            covariates.append(covariate)
            continue
        data[f'{covariate}_early'] = data[covariate] * (1 - data['late_period'])
        data[f'{covariate}_late'] = data[covariate] * data['late_period']
        covariates += [f'{covariate}_early', f'{covariate}_late']

    fitter = CoxTimeVaryingFitter()
    fitter.fit(data[['patient', 'start', 'stop', 'event'] + covariates], id_col='patient', event_col='event',
               start_col='start', stop_col='stop')
    params = fitter.params_
    variance = pd.DataFrame(np.asarray(fitter.variance_matrix_), index=params.index, columns=params.index)

    rows = []
    for covariate in PIECEWISE_COVARIATES:
        early, late = f'{covariate}_early', f'{covariate}_late'
        log_ratio = params[late] - params[early]
        se = np.sqrt(variance.loc[late, late] + variance.loc[early, early] - 2 * variance.loc[late, early])
        rows.append((
            covariate,
            np.exp(params[early]), np.exp(params[late]),
            np.exp(log_ratio), np.exp(log_ratio - Z_95 * se), np.exp(log_ratio + Z_95 * se),
            2 * stats.norm.sf(abs(log_ratio / se)),
        ))

    events_early = int(data.loc[data['late_period'] == 0, 'event'].sum())
    events_late = int(data.loc[data['late_period'] == 1, 'event'].sum())
    columns = ['covariate', 'HR_early', 'HR_late', 'ratio_late_vs_early', 'lower', 'upper', 'p']
    return pd.DataFrame(rows, columns=columns).assign(events_early=events_early, events_late=events_late)


def fine_gray_table(dataset: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    result = cr.fine_gray(dataset, 'time', 'event', covariates)
    table = hazard_ratios(result.summary, 'Fine-Gray')
    return table.rename(columns={'HR': 'sHR'})


@dataclass(frozen=True)
class RidgeResult:
    table: pd.DataFrame
    penalizer: float
    dropped_sparse: list[str]


def _is_sparse(values: pd.Series) -> bool:
    """Binary covariate with too few patients in either level, e.g. clopidogrel 1/313."""
    if values.nunique() > 2:
        return False
    return values.value_counts().min() < MIN_PATIENTS_PER_LEVEL or values.nunique() < 2


def ridge_model(dataset: pd.DataFrame, covariates: list[str]) -> RidgeResult:
    """Ridge cause-specific Cox; penalty by 5-fold CV, 95% CI by bootstrap percentiles."""
    sparse = [c for c in covariates if _is_sparse(dataset[c])]
    covariates = [c for c in covariates if c not in sparse]
    data = _event_indicator(dataset, Event.DCI)[['time', 'is_event'] + covariates]

    scores = {}
    for penalizer in RIDGE_PENALIZERS:
        folds = k_fold_cross_validation(CoxPHFitter(penalizer=penalizer), data, duration_col='time',
                                        event_col='is_event', k=CV_FOLDS, seed=RANDOM_SEED)
        scores[penalizer] = np.mean(folds)
    penalizer = max(scores, key=scores.get)

    estimate = CoxPHFitter(penalizer=penalizer).fit(data, duration_col='time', event_col='is_event').params_

    rng = np.random.default_rng(RANDOM_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = data.iloc[rng.integers(0, len(data), len(data))]
        if (sample[covariates].nunique() < 2).any():
            continue
        boot.append(CoxPHFitter(penalizer=penalizer).fit(sample, duration_col='time', event_col='is_event').params_)
    boot = pd.DataFrame(boot)

    table = pd.DataFrame({
        'model': f'ridge (penalizer {penalizer:g})',
        'covariate': estimate.index,
        'HR': np.exp(estimate.values),
        'lower': np.exp(boot.quantile(CI_PERCENTILES[0] / 100).reindex(estimate.index).values),
        'upper': np.exp(boot.quantile(CI_PERCENTILES[1] / 100).reindex(estimate.index).values),
        'bootstrap_fits': len(boot),
    })
    return RidgeResult(table=table, penalizer=penalizer, dropped_sparse=sparse)


def _absolute_risk_models(dataset: pd.DataFrame, covariates: list[str]) -> tuple[CoxPHFitter, CoxPHFitter]:
    dci_model = fit_cause_specific(dataset, covariates, Event.DCI)
    death_model = fit_cause_specific(dataset, covariates, Event.DEATH, penalizer=DEATH_MODEL_PENALIZER)
    return dci_model, death_model


def _performance(dataset: pd.DataFrame, risk: np.ndarray) -> dict[str, float]:
    time, event = dataset['time'].to_numpy(), dataset['event'].to_numpy()
    return {
        'calibration_slope': cr.calibration_slope(time, event, risk, HORIZON),
        'brier_day21': cr.brier_score(time, event, risk, HORIZON),
        'c_index': cr.c_index(time, event, risk, HORIZON),
    }


@dataclass(frozen=True)
class ModelCheck:
    metrics: pd.DataFrame
    calibration: pd.DataFrame


def model_check(dataset: pd.DataFrame, covariates: list[str]) -> ModelCheck:
    """Apparent and optimism-corrected (bootstrap) performance at day 21."""
    dci_model, death_model = _absolute_risk_models(dataset, covariates)
    risk = cr.cause_specific_risk(dci_model, death_model, dataset[covariates], HORIZON)
    apparent = _performance(dataset, risk)

    rng = np.random.default_rng(RANDOM_SEED)
    optimism, failed = [], 0
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = dataset.iloc[rng.integers(0, len(dataset), len(dataset))].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boot_dci, boot_death = _absolute_risk_models(sample, covariates)
                on_sample = _performance(sample, cr.cause_specific_risk(boot_dci, boot_death, sample[covariates], HORIZON))
                on_original = _performance(dataset, cr.cause_specific_risk(boot_dci, boot_death, dataset[covariates], HORIZON))
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        optimism.append({k: on_sample[k] - on_original[k] for k in apparent})

    mean_optimism = pd.DataFrame(optimism).mean()
    metrics = pd.DataFrame({
        'metric': list(apparent),
        'apparent': list(apparent.values()),
        'optimism': [mean_optimism[k] for k in apparent],
    })
    metrics['corrected'] = metrics['apparent'] - metrics['optimism']
    metrics['bootstrap_fits'] = len(optimism)
    metrics['bootstrap_failed'] = failed

    calibration = cr.calibration_by_group(dataset['time'], dataset['event'], risk, HORIZON, CALIBRATION_GROUPS)
    return ModelCheck(metrics=metrics, calibration=calibration)


def cumulative_incidence(dataset: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    """Aalen-Johansen curves of late DCI and competing death, overall or by group."""
    if group_column is None:
        return cr.aalen_johansen(dataset['time'], dataset['event']).assign(group='all')

    curves = []
    for value, group in dataset.dropna(subset=[group_column]).groupby(group_column):
        curves.append(cr.aalen_johansen(group['time'], group['event']).assign(group=f'{group_column}={value:g}'))
    return pd.concat(curves, ignore_index=True)
