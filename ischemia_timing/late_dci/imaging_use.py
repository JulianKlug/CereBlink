"""Stability of imaging use over calendar time (reviewer: management and modalities changed 2009-2023).

Two questions:
    - perfusion CTs per patient vs calendar year (all patients)
    - availability of each perfusion metric at DCI diagnosis vs calendar year (verified DCI only;
      metrics are only recorded for DCI patients)
"""
from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .cohort import PERFUSION_METRICS

CONFIDENCE = 0.95
Z_95 = stats.norm.ppf(0.5 + CONFIDENCE / 2)

ANY_PCT = 'any_pct'
HOSPITAL_DAYS = 'hospital_days'

# Year centred for numerical stability, e.g. 2019 -> 3; per-year estimates unchanged
REFERENCE_YEAR = 2016
DCI_LABELS = {True: 'DCI', False: 'no DCI'}


class Exposure(Enum):
    NONE = 'none'
    HOSPITAL_DAYS = 'hospital days'


# (model name, formula, exposure); hospital days as exposure -> pCT rate per hospital day
PCT_MODELS = [
    ('crude', 'n_pct ~ year', Exposure.NONE),
    ('+ DCI status, per hospital day', 'n_pct ~ year + dci', Exposure.HOSPITAL_DAYS),
]


def _available_for_pct(patients: pd.DataFrame) -> pd.DataFrame:
    # Negative length of stay = date error (n=1), excluded
    has_data = patients['n_pct'].notna() & patients['year'].notna() & (patients['t_discharge'] >= 0)

    # Admission day counts, e.g. death on the day of admission -> 1 hospital day
    return patients[has_data].assign(
        dci=patients['dci'].astype(int),
        any_pct=(patients['n_pct'] > 0).astype(int),
        hospital_days=patients['t_discharge'] + 1,
    )


def _centred(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(year=data['year'] - REFERENCE_YEAR)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    # t-interval, e.g. mean 2.1 (1.6-2.6); single patient -> no interval
    mean = values.mean()
    if len(values) < 2:
        return mean, np.nan, np.nan
    half_width = stats.t.ppf(0.5 + CONFIDENCE / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half_width, mean + half_width


def pct_by_year(patients: pd.DataFrame) -> pd.DataFrame:
    """pCTs per patient by calendar year, overall and by DCI status."""
    data = _available_for_pct(patients)
    groups = [('all', data)] + [(DCI_LABELS[bool(is_dci)], group) for is_dci, group in data.groupby('dci')]

    rows = []
    for label, group in groups:
        for year, year_group in group.groupby('year'):
            mean, lower, upper = _mean_ci(year_group['n_pct'])
            rows.append({
                'group': label, 'year': int(year), 'n': len(year_group),
                'mean_pct': mean, 'lower': lower, 'upper': upper,
                'median_pct': year_group['n_pct'].median(),
                'any_pct_percent': 100 * year_group[ANY_PCT].mean(),
            })
    return pd.DataFrame(rows)


def _trend_row(model: str, outcome: str, estimate_name: str, fit, n: int, n_events: int) -> dict:
    coef, se = fit.params['year'], fit.bse['year']
    return {
        'outcome': outcome, 'model': model, 'n': n, 'n_positive': n_events, 'estimate': estimate_name,
        'per_year': np.exp(coef), 'lower': np.exp(coef - Z_95 * se), 'upper': np.exp(coef + Z_95 * se),
        'p': fit.pvalues['year'],
    }


def pct_trend(patients: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year trend in pCT use.

    Count: negative binomial, IRR per year (crude; then + DCI status with hospital days as exposure).
    Any pCT: logistic, OR per year.
    """
    data = _centred(_available_for_pct(patients))
    rows = []

    for model, formula, exposure_type in PCT_MODELS:
        exposure = data[HOSPITAL_DAYS] if exposure_type == Exposure.HOSPITAL_DAYS else None
        fit = smf.negativebinomial(formula, data, exposure=exposure).fit(disp=False)
        rows.append(_trend_row(model, 'number of pCTs', 'IRR', fit, len(data), int((data['n_pct'] > 0).sum())))

    fit = smf.logit('any_pct ~ year', data).fit(disp=False)
    rows.append(_trend_row('crude', '>= 1 pCT', 'OR', fit, len(data), int(data[ANY_PCT].sum())))
    fit = smf.logit('any_pct ~ year + dci', data).fit(disp=False)
    rows.append(_trend_row('+ DCI status', '>= 1 pCT', 'OR', fit, len(data), int(data[ANY_PCT].sum())))
    return pd.DataFrame(rows)


def _dci_patients(patients: pd.DataFrame) -> pd.DataFrame:
    return patients[patients['dci'] & patients['year'].notna()]


def perfusion_availability_by_year(patients: pd.DataFrame) -> pd.DataFrame:
    """Per year and metric: DCI patients with the metric reported / all DCI patients."""
    dci = _dci_patients(patients)
    rows = []
    for year, group in dci.groupby('year'):
        for metric in PERFUSION_METRICS:
            available = int(group[f'{metric}_available'].sum())
            rows.append({'year': int(year), 'metric': metric, 'n_available': available, 'n_dci': len(group),
                         'percent': 100 * available / len(group)})
    return pd.DataFrame(rows)


def perfusion_availability_trend(patients: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression of metric availability on calendar year among DCI patients, OR per year."""
    dci = _dci_patients(patients)
    rows = []
    for metric in PERFUSION_METRICS:
        data = _centred(pd.DataFrame({'available': dci[f'{metric}_available'].astype(int), 'year': dci['year']}))

        # Quasi-separation (e.g. a metric reported every year) -> estimate unreliable, flagged
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = smf.glm('available ~ year', data, family=sm.families.Binomial()).fit()
        row = _trend_row('crude', metric, 'OR', fit, len(data), int(data['available'].sum()))
        row['warning'] = '; '.join(sorted({str(w.message) for w in caught}))
        rows.append(row)
    return pd.DataFrame(rows)


def dci_trend(patients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verified DCI by calendar year (ascertainment stability), logistic OR per year."""
    data = _centred(patients[patients['year'].notna()].assign(dci=patients['dci'].astype(int)))
    fit = smf.logit('dci ~ year', data).fit(disp=False)

    by_year = patients.groupby('year')['dci'].agg(['sum', 'size'])
    by_year = by_year.reset_index().rename(columns={'sum': 'n_dci', 'size': 'n'}).astype(int)
    by_year['percent'] = 100 * by_year['n_dci'] / by_year['n']

    trend = pd.DataFrame([_trend_row('crude', 'verified DCI', 'OR', fit, len(data), int(data['dci'].sum()))])
    return by_year, trend
