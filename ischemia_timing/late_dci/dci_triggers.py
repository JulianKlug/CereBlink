"""Circumstances of DCI diagnosis (editor critical 1: clinical vs surrogate ascertainment).

Verified DCI patients only; each item reported as n yes / n recorded, blanks counted separately.
Monitoring triggers: blank = device not in use -> not a trigger, e.g. no PtiO2 probe -> 'no'.
TCD is done in every patient, so a blank TCD cell = missing.

    all DCI                  clinically assessable, pCT verification
      |-- clinically assessable   decreased consciousness, focal neurological signs
      |-- pCT verified            what triggered the pCT (not mutually exclusive)

Trigger source of a pCT-verified diagnosis, from recorded findings:
    clinical    decreased consciousness, focal signs, unexplained DoC / deficit, delirium
    monitoring  ICP, TCD, PtiO2, microdialysis, NIRS
e.g. focal signs + suspect TCD -> 'clinical and monitoring'; nothing recorded -> 'none recorded'.

Ascertainment analyses (late_dci_analysis/analysis_plan_dci_ascertainment.md):
    1 device_use        devices in use at diagnosis, positive among in use
    2 onset_by_source   onset day by any monitoring trigger / by assessability
    3 source_by_year    ascertainment by calendar year; year effect on onset with / without monitoring
    4 late_tail         onset after day 14 vs earlier
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from .cohort import CLINICAL_ASSESSABILITY, CLINICAL_SIGNS, PCT_TRIGGERS, PCT_VERIFICATION, RECURRENCE

YES = 1
NO = 0
CONFIDENCE = 0.95
CI_METHOD = 'wilson'

# Triggers requiring a monitoring device or study; clinical triggers keep blanks as missing
MONITORING_TRIGGERS = ['raised_icp', 'suspect_TCD', 'decreased_ptio2', 'suspect_microdialysis', 'decreased_NIRS']
# Devices used in every patient by protocol: blank = missing, not 'not in use'
UNIVERSAL_DEVICES = ['suspect_TCD']
BLANK_AS_NO = [trigger for trigger in MONITORING_TRIGGERS if trigger not in UNIVERSAL_DEVICES]
CLINICAL_TRIGGERS = CLINICAL_SIGNS + [trigger for trigger in PCT_TRIGGERS if trigger not in MONITORING_TRIGGERS]

# Decreased consciousness or focal signs, one set in trigger combinations
CLINICAL_DETERIORATION = 'clinical deterioration'
COMBINATION_SETS = [CLINICAL_DETERIORATION] + PCT_TRIGGERS

# Codes -> path labels, e.g. clinically_assessable 0 -> 'not assessable', blank -> 'not recorded'
ASSESSABILITY_LABELS = {1: 'assessable', 0: 'not assessable'}
PCT_LABELS = {1: 'pCT', 0: 'no pCT'}
NOT_RECORDED = 'not recorded'
NO_PCT = 'no pCT'

Z_95 = stats.norm.ppf(0.5 + CONFIDENCE / 2)
MEDIAN = 0.5
BOOTSTRAP_SAMPLES = 2000
SEED = 2016

# Classic vasospasm window ends on day 14; fixed a priori
LATE_TAIL_DAY = 14.0
TAIL_QUANTILE = 0.95

# Year centred as in imaging_use, e.g. 2019 -> 3
REFERENCE_YEAR = 2016
PERIOD_EDGES = [2010, 2014, 2018, np.inf]
PERIOD_LABELS = ['2011-14', '2015-18', '2019-']

# Devices with enough patients for a year trend; others descriptive only
TREND_DEVICES = ['raised_icp']
ADJUSTMENT = 'poor_wfns + age + year_c'


class TriggerSource(Enum):
    CLINICAL_ONLY = 'clinical only'
    MONITORING_ONLY = 'monitoring only'
    BOTH = 'clinical and monitoring'
    NONE = 'none recorded'


class Population(Enum):
    ALL_DCI = 'all DCI'
    CLINICALLY_ASSESSABLE = 'clinically assessable'
    PCT_VERIFIED = 'pCT verified'


def _members(dci: pd.DataFrame, population: Population) -> pd.DataFrame:
    if population == Population.CLINICALLY_ASSESSABLE:
        return dci[dci[CLINICAL_ASSESSABILITY] == YES]

    if population == Population.PCT_VERIFIED:
        return dci[dci[PCT_VERIFICATION] == YES]

    return dci


def _frequency(values: pd.Series, population: Population) -> dict:
    # Percent of recorded values, e.g. 45 yes of 111 recorded (2 blank) -> 40.5%
    n_blank_as_no = int(values.isna().sum()) if values.name in BLANK_AS_NO else 0
    if n_blank_as_no:
        values = values.fillna(NO)

    recorded = values.dropna()
    n_yes = int((recorded == YES).sum())
    row = {'population': population.value, 'n_population': len(values), 'item': values.name,
           'n_yes': n_yes, 'n_recorded': len(recorded), 'n_missing': int(values.isna().sum()),
           'n_blank_as_no': n_blank_as_no}

    if recorded.empty:
        return row | {'percent': float('nan'), 'lower': float('nan'), 'upper': float('nan')}

    lower, upper = proportion_confint(n_yes, len(recorded), alpha=1 - CONFIDENCE, method=CI_METHOD)
    return row | {'percent': 100 * n_yes / len(recorded), 'lower': 100 * lower, 'upper': 100 * upper}


def diagnosis_circumstances(patients: pd.DataFrame) -> pd.DataFrame:
    """One row per (population, item): frequency among verified DCI patients."""
    dci = patients[patients['dci']]
    items = [
        (Population.ALL_DCI, [CLINICAL_ASSESSABILITY, PCT_VERIFICATION]),
        (Population.CLINICALLY_ASSESSABLE, CLINICAL_SIGNS),
        (Population.PCT_VERIFIED, PCT_TRIGGERS),
    ]

    rows = []
    for population, fields in items:
        members = _members(dci, population)
        rows += [_frequency(members[field], population) for field in fields]
    return pd.DataFrame(rows)


def _any_yes(dci: pd.DataFrame, fields: list[str]) -> pd.Series:
    return (dci[fields] == YES).any(axis=1)


def _trigger_source(dci: pd.DataFrame) -> pd.Series:
    clinical = _any_yes(dci, CLINICAL_TRIGGERS)
    monitoring = _any_yes(dci, MONITORING_TRIGGERS)
    source = pd.Series(TriggerSource.NONE.value, index=dci.index)
    source[clinical & monitoring] = TriggerSource.BOTH.value
    source[clinical & ~monitoring] = TriggerSource.CLINICAL_ONLY.value
    source[~clinical & monitoring] = TriggerSource.MONITORING_ONLY.value
    return source


def diagnostic_paths(patients: pd.DataFrame) -> pd.DataFrame:
    """Verified DCI patients per (assessability, pCT verification, trigger source); 'no pCT' has no source."""
    dci = patients[patients['dci']]
    pct = dci[PCT_VERIFICATION].map(PCT_LABELS).fillna(NOT_RECORDED)
    paths = pd.DataFrame({
        'assessability': dci[CLINICAL_ASSESSABILITY].map(ASSESSABILITY_LABELS).fillna(NOT_RECORDED),
        'pct': pct,
        'source': _trigger_source(dci).where(pct == PCT_LABELS[YES], NO_PCT),
    })
    return paths.value_counts().rename('n').reset_index()


def trigger_combinations(patients: pd.DataFrame) -> pd.DataFrame:
    """pCT-verified patients per combination of recorded triggers (one boolean column per set), largest first."""
    verified = _members(patients[patients['dci']], Population.PCT_VERIFIED)
    sets = pd.DataFrame({field: verified[field] == YES for field in PCT_TRIGGERS})
    sets.insert(0, CLINICAL_DETERIORATION, _any_yes(verified, CLINICAL_SIGNS))
    return sets.value_counts().rename('n').reset_index()


# Ascertainment analyses ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OnsetBySource:
    groups: pd.DataFrame       # median (IQR) onset per group
    comparisons: pd.DataFrame  # exposed - reference, crude and adjusted


@dataclass(frozen=True)
class SourceByYear:
    by_period: pd.DataFrame
    trend: pd.DataFrame
    year_on_onset: pd.DataFrame  # year coefficient with / without monitoring exposure


@dataclass(frozen=True)
class LateTail:
    comparison: pd.DataFrame
    distribution: pd.DataFrame


def _ascertainment_table(patients: pd.DataFrame) -> pd.DataFrame:
    """Verified DCI with derived exposures, e.g. source 'monitoring only' -> any_monitoring 1."""
    dci = patients[patients['dci']].copy()
    source = _trigger_source(dci)
    monitoring_sources = [TriggerSource.MONITORING_ONLY.value, TriggerSource.BOTH.value]

    dci['source'] = source
    dci['pct_verified'] = dci[PCT_VERIFICATION] == YES
    dci['any_monitoring'] = source.isin(monitoring_sources).astype(int)
    dci['not_assessable'] = (dci[CLINICAL_ASSESSABILITY] == NO).astype(int)
    dci['year_c'] = dci['year'] - REFERENCE_YEAR
    dci['period'] = pd.cut(dci['year'], bins=PERIOD_EDGES, labels=PERIOD_LABELS)
    return dci


def _monitoring_vs_clinical(dci: pd.DataFrame) -> pd.DataFrame:
    # pCT verified, any monitoring trigger or clinical only; 'none recorded' left out
    verified = dci[dci['pct_verified']]
    return verified[(verified['any_monitoring'] == 1) | (verified['source'] == TriggerSource.CLINICAL_ONLY.value)]


def _proportion(n_yes: int, n: int) -> dict:
    if n == 0:
        return {'percent': np.nan, 'lower': np.nan, 'upper': np.nan}
    lower, upper = proportion_confint(n_yes, n, alpha=1 - CONFIDENCE, method=CI_METHOD)
    return {'percent': 100 * n_yes / n, 'lower': 100 * lower, 'upper': 100 * upper}


def _fisher(a_yes: int, a_n: int, b_yes: int, b_n: int) -> float:
    if min(a_n, b_n) == 0:
        return np.nan
    return stats.fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])[1]


def device_use(patients: pd.DataFrame) -> pd.DataFrame:
    """Per device, among verified DCI: in use at diagnosis, positive among recorded, by assessability.

    In use = cell filled, except universal devices (TCD): in use in all, blank = missing,
    e.g. TCD 95 recorded of 112 -> in use 112, positive among 95.
    """
    dci = _ascertainment_table(patients)
    assessable = dci[dci[CLINICAL_ASSESSABILITY] == YES]
    not_assessable = dci[dci[CLINICAL_ASSESSABILITY] == NO]

    rows = []
    for device in MONITORING_TRIGGERS:
        universal = device in UNIVERSAL_DEVICES
        n_recorded, n_positive = int(dci[device].notna().sum()), int((dci[device] == YES).sum())
        n_in_use = len(dci) if universal else n_recorded
        use_assessable = len(assessable) if universal else int(assessable[device].notna().sum())
        use_not = len(not_assessable) if universal else int(not_assessable[device].notna().sum())

        row = {'device': device, 'n_dci': len(dci), 'n_in_use': n_in_use}
        row |= {f'in_use_{k}': v for k, v in _proportion(n_in_use, len(dci)).items()}
        row |= {'n_recorded': n_recorded, 'n_positive': n_positive}
        row |= {f'positive_{k}': v for k, v in _proportion(n_positive, n_recorded).items()}
        row |= {'in_use_assessable': f'{use_assessable}/{len(assessable)}',
                'in_use_not_assessable': f'{use_not}/{len(not_assessable)}',
                'p_assessability': np.nan if universal else _fisher(use_not, len(not_assessable), use_assessable, len(assessable))}
        rows.append(row)
    return pd.DataFrame(rows)


def hodges_lehmann(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Median of all x - y differences with the distribution-free (Moses) 95% CI."""
    differences = np.sort(np.subtract.outer(np.asarray(x), np.asarray(y)).ravel())
    m, n = len(x), len(y)

    # Rank of the lower limit among the m * n differences, normal approximation to the U distribution
    k = max(int(np.floor(m * n / 2 - Z_95 * np.sqrt(m * n * (m + n + 1) / 12))), 0)
    return float(np.median(differences)), float(differences[k]), float(differences[m * n - 1 - k])


def _median_regression(data: pd.DataFrame, formula: str, term: str) -> tuple[float, float, float]:
    """Coefficient of `term` in a median regression, percentile bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    with warnings.catch_warnings():
        # Bootstrap resamples with tied onsets trigger iteration / singularity warnings
        warnings.simplefilter('ignore')
        estimate = smf.quantreg(formula, data).fit(q=MEDIAN).params[term]
        replicates = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sample = data.iloc[rng.integers(0, len(data), len(data))]
            replicates.append(smf.quantreg(formula, sample).fit(q=MEDIAN).params[term])

    alpha = (1 - CONFIDENCE) / 2
    lower, upper = np.nanquantile(replicates, [alpha, 1 - alpha])
    return float(estimate), float(lower), float(upper)


def _onset_summary(label: str, onset: pd.Series) -> dict:
    return {'group': label, 'n': len(onset), 'median': onset.median(),
            'q1': onset.quantile(0.25), 'q3': onset.quantile(0.75)}


def _comparison(label: str, data: pd.DataFrame, exposure: str) -> dict:
    exposed, reference = data.loc[data[exposure] == 1, 't_dci'], data.loc[data[exposure] == 0, 't_dci']
    estimate, lower, upper = hodges_lehmann(exposed, reference)
    adjusted = data.dropna(subset=['poor_wfns', 'age', 'year_c'])
    adj_estimate, adj_lower, adj_upper = _median_regression(adjusted, f't_dci ~ {exposure} + {ADJUSTMENT}', exposure)
    return {
        'comparison': label, 'n_exposed': len(exposed), 'n_reference': len(reference),
        'p_mann_whitney': stats.mannwhitneyu(exposed, reference).pvalue,
        'hl_difference': estimate, 'hl_lower': lower, 'hl_upper': upper,
        'n_adjusted': len(adjusted), 'adjusted_difference': adj_estimate,
        'adjusted_lower': adj_lower, 'adjusted_upper': adj_upper,
    }


def onset_by_source(patients: pd.DataFrame) -> OnsetBySource:
    """Onset day (days from ictus) by trigger source and by assessability; differences exposed - reference."""
    dci = _ascertainment_table(patients)
    verified = dci[dci['pct_verified']]

    groups = [_onset_summary(f'pCT verified: {source}', verified.loc[verified['source'] == source, 't_dci'])
              for source in [s.value for s in TriggerSource]]
    groups += [_onset_summary(label, dci.loc[dci[CLINICAL_ASSESSABILITY] == code, 't_dci'])
               for code, label in ASSESSABILITY_LABELS.items()]

    assessability_known = dci[dci[CLINICAL_ASSESSABILITY].notna()]
    comparisons = [
        _comparison('any monitoring vs clinical only (pCT verified)', _monitoring_vs_clinical(dci), 'any_monitoring'),
        _comparison('not assessable vs assessable (all DCI)', assessability_known, 'not_assessable'),
    ]
    return OnsetBySource(groups=pd.DataFrame(groups), comparisons=pd.DataFrame(comparisons))


def _year_outcomes(dci: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    # (outcome, population, 0/1 outcome), e.g. TCD in use among all DCI
    verified = dci[dci['pct_verified']]
    pct_known = dci[dci[PCT_VERIFICATION].notna()]
    assessability_known = dci[dci[CLINICAL_ASSESSABILITY].notna()]
    outcomes = [
        ('any monitoring trigger (pCT verified)', verified, verified['any_monitoring']),
        ('pCT verification (recorded)', pct_known, pct_known['pct_verified'].astype(int)),
        ('clinically assessable (recorded)', assessability_known, (assessability_known[CLINICAL_ASSESSABILITY] == YES).astype(int)),
    ]
    outcomes += [(f'{device} in use (all DCI)', dci, dci[device].notna().astype(int)) for device in BLANK_AS_NO]
    return outcomes


def source_by_year(patients: pd.DataFrame) -> SourceByYear:
    """Ascertainment by period and per calendar year (logistic, OR per year); year effect on onset."""
    dci = _ascertainment_table(patients)
    period_rows, trend_rows = [], []

    for name, population, outcome in _year_outcomes(dci):
        for period in PERIOD_LABELS:
            in_period = population['period'] == period
            n, n_yes = int(in_period.sum()), int(outcome[in_period].sum())
            period_rows.append({'outcome': name, 'period': period, 'n_yes': n_yes, 'n': n} | _proportion(n_yes, n))

        if name.split(' ')[0] in MONITORING_TRIGGERS and name.split(' ')[0] not in TREND_DEVICES:
            continue
        data = pd.DataFrame({'outcome': outcome, 'year_c': population['year_c']})
        fit = smf.glm('outcome ~ year_c', data, family=sm.families.Binomial()).fit()
        coef, se = fit.params['year_c'], fit.bse['year_c']
        trend_rows.append({'outcome': name, 'n': len(data), 'n_yes': int(outcome.sum()), 'or_per_year': np.exp(coef),
                           'lower': np.exp(coef - Z_95 * se), 'upper': np.exp(coef + Z_95 * se),
                           'p': fit.pvalues['year_c']})

    # Same sample for both models, so a change in the year coefficient reflects the monitoring term only
    sample = _monitoring_vs_clinical(dci).dropna(subset=['poor_wfns', 'age', 'year_c'])
    year_rows = []
    for model, formula in [('without monitoring', f't_dci ~ {ADJUSTMENT}'),
                           ('with monitoring', f't_dci ~ any_monitoring + {ADJUSTMENT}')]:
        estimate, lower, upper = _median_regression(sample, formula, 'year_c')
        year_rows.append({'model': model, 'n': len(sample), 'days_per_year': estimate, 'lower': lower, 'upper': upper})

    return SourceByYear(by_period=pd.DataFrame(period_rows), trend=pd.DataFrame(trend_rows),
                        year_on_onset=pd.DataFrame(year_rows))


def _tail_distribution(label: str, onset: pd.Series) -> dict:
    n_late = int((onset > LATE_TAIL_DAY).sum())
    return _onset_summary(label, onset) | {'p95': onset.quantile(TAIL_QUANTILE), 'n_late': n_late,
                                           'percent_late': 100 * n_late / len(onset) if len(onset) else np.nan}


def late_tail(patients: pd.DataFrame) -> LateTail:
    """Onset after day 14 vs earlier: ascertainment characteristics, and the tail in clinically detected DCI."""
    dci = _ascertainment_table(patients)
    dci['late'] = dci['t_dci'] > LATE_TAIL_DAY
    verified = dci[dci['pct_verified']]
    none_recorded = (verified['source'] == TriggerSource.NONE.value).astype(int)

    characteristics = [
        ('any monitoring trigger', 'pCT verified', verified, verified['any_monitoring']),
        ('none recorded', 'pCT verified', verified, none_recorded),
        ('clinically assessable', 'recorded', dci[dci[CLINICAL_ASSESSABILITY].notna()], dci[CLINICAL_ASSESSABILITY] == YES),
        ('pCT verification', 'recorded', dci[dci[PCT_VERIFICATION].notna()], dci['pct_verified']),
        ('DCI recurrence', 'recorded', dci[dci[RECURRENCE].notna()], dci[RECURRENCE] == YES),
    ]
    rows = []
    for name, population_label, population, flag in characteristics:
        flag = flag.loc[population.index].astype(int)
        late, early = population['late'], ~population['late']
        late_yes, late_n, early_yes, early_n = int(flag[late].sum()), int(late.sum()), int(flag[early].sum()), int(early.sum())
        rows.append({'characteristic': name, 'population': population_label,
                     'late': f'{late_yes}/{late_n}', 'late_percent': 100 * late_yes / late_n if late_n else np.nan,
                     'early': f'{early_yes}/{early_n}', 'early_percent': 100 * early_yes / early_n if early_n else np.nan,
                     'p_fisher': _fisher(late_yes, late_n, early_yes, early_n)})

    clinically_detected = (dci[CLINICAL_ASSESSABILITY] == YES) & (dci['source'] == TriggerSource.CLINICAL_ONLY.value)
    distribution = [
        _tail_distribution('all DCI', dci['t_dci'].dropna()),
        _tail_distribution('clinically detected, assessable', dci.loc[clinically_detected, 't_dci'].dropna()),
        _tail_distribution('any monitoring trigger', dci.loc[dci['any_monitoring'] == 1, 't_dci'].dropna()),
    ]
    return LateTail(comparison=pd.DataFrame(rows), distribution=pd.DataFrame(distribution))


def late_cases(patients: pd.DataFrame) -> pd.Index:
    """Row labels of DCI with onset after day 14, for internal chart verification only."""
    dci = patients[patients['dci']]
    return dci.index[dci['t_dci'] > LATE_TAIL_DAY]
