"""DCI timing vs functional outcome (mRS at follow-up) and DCI-related infarction, among DCI patients.

Unadjusted: Mann-Whitney U of DCI day between outcome groups.
Adjusted: DCI day + age, sex, WFNS, modified Fisher; effect = OR per day from haemorrhage (> 1 = later DCI, worse).

    ModelSpec.SUBMITTED      reproduces the submitted manuscript: probit on mRS > 2 (missing mRS counted
                             as > 2) and on infarction; exp(coefficient) reported as 'OR'
    ModelSpec.ORDINAL_LOGIT  as described in the Methods: ordinal logistic on mRS 0-6 (missing excluded),
                             binary logistic on infarction
    ModelSpec.BINARY_LOGIT   comparison: binary logistic on mRS > 2 (missing excluded)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .data_sources import RawSources
from .figure1 import DCI, event_days
from .table1 import CLINICAL_ASSESSABILITY, DEAD_MRS, FOLLOW_UP_DEATH, FOLLOW_UP_VISITS, YES, YearFilter, select_registry

DCI_DAY = 'dci_day'
COVARIATES = ['age', 'male', 'wfns', 'fisher']
FAVOURABLE_MAX_MRS = 2
CONFIDENCE = 0.95
DAYS_PER_MONTH = 30.44
FIT_MAX_ITERATIONS = 10000
BEST_CASE_MRS = 0
SURVIVAL_LANDMARK_DAYS = [14, 21]
MAX_MRS_THRESHOLD = 5  # mRS > 5 = death


class ModelSpec(Enum):
    SUBMITTED = 'submitted (probit)'
    ORDINAL_LOGIT = 'logistic'  # ordinal for mRS 0-6, binary for infarction
    BINARY_LOGIT = 'logistic, mRS > 2'


class Outcome(Enum):
    MRS = 'mRS at follow-up'
    INFARCTION = 'DCI-related infarction'


class _Family(Enum):
    PROBIT_ORDERED = 'probit'   # submitted models
    LOGIT_ORDERED = 'logit'     # mRS 0-6
    LOGIT = 'binary logit'      # infarction, missingness


class MissingMrs(Enum):
    UNFAVOURABLE = 'counted as mRS > 2'
    EXCLUDED = 'excluded'


def _date(value) -> pd.Timestamp:
    # Follow-up dates, e.g. datetime, '14/08/2023', '20.07.2022'; ' ', '0', 'NA' -> NaT
    if isinstance(value, (dt.datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if pd.isna(value) or not isinstance(value, str):
        return pd.NaT
    return pd.to_datetime(value.strip(), dayfirst=True, errors='coerce')


def build_outcome_data(sources: RawSources, year_filter: YearFilter, first_year: int) -> pd.DataFrame:
    """One row per DCI patient with a DCI onset day: outcomes, covariates and sensitivity-analysis fields."""
    registry = select_registry(sources, year_filter, first_year)
    admission = pd.to_datetime(registry['Date_admission'])
    ictus = pd.to_datetime(registry['Date_Ictus'].map(_date)).fillna(admission)
    death_date = pd.to_datetime(registry['Date_Death'].map(_date)).where(registry['Death'] == YES)
    follow_up_date = pd.to_datetime(registry['Date_FU_used'].map(_date), errors='coerce')

    data = pd.DataFrame({
        DCI_DAY: event_days(registry)[DCI],
        'mrs': registry['mRS_FU_1y'],
        # Extreme cases for missing mRS: all best, all worst
        'mrs_missing_0': registry['mRS_FU_1y'].fillna(BEST_CASE_MRS),
        'mrs_missing_6': registry['mRS_FU_1y'].fillna(DEAD_MRS),
        'death_day': (death_date - ictus).dt.days,
        CLINICAL_ASSESSABILITY: registry[CLINICAL_ASSESSABILITY],
        'mrs_discharge': registry['mRS_discharge'],
        'infarction': pd.to_numeric(registry['DCI_infarct'], errors='coerce').astype(float),  # bool -> 0/1
        'age': registry['Age'],
        'male': registry['male'],
        'wfns': pd.to_numeric(registry['WFNS'], errors='coerce'),
        'fisher': registry['Fisher_Score'],
        'year': admission.dt.year,
        'in_hospital_death': registry['mRS_discharge'] == DEAD_MRS,
        'ictus_documented': registry['Date_Ictus'].notna(),
        'mrs_source': registry['mRS_FU_source'],
        # Interval as in the manuscript (admission to follow-up visit), e.g. 7.5 months
        'follow_up_months': (follow_up_date - admission).dt.days / DAYS_PER_MONTH,
    }, index=registry.index)
    return data[data[DCI_DAY].notna()]


@dataclass(frozen=True)
class _Fit:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    n: int
    llf: float
    aic: float


def _fit(y: pd.Series, X: pd.DataFrame, family: _Family) -> _Fit:
    if family == _Family.LOGIT:
        result = sm.Logit(y, sm.add_constant(X)).fit(disp=False, maxiter=FIT_MAX_ITERATIONS)
    else:
        result = OrderedModel(y, X, distr=family.value).fit(method='bfgs', disp=False, maxiter=FIT_MAX_ITERATIONS)
    return _Fit(result.params, result.bse, result.pvalues, int(result.nobs), result.llf, result.aic)


def _odds_ratio(fit: _Fit, covariate: str) -> dict:
    # Wald interval on the log-odds scale, e.g. OR 0.91 (0.83-1.00)
    z = stats.norm.ppf(0.5 + CONFIDENCE / 2)
    coef, se = fit.params[covariate], fit.bse[covariate]
    return {'OR_per_day': np.exp(coef), 'lower': np.exp(coef - z * se), 'upper': np.exp(coef + z * se),
            'p': fit.pvalues[covariate]}


def _model_frame(data: pd.DataFrame, outcome: Outcome, spec: ModelSpec, missing: MissingMrs,
                 mrs_column: str) -> tuple[pd.Series, _Family]:
    """Dependent variable and model family for an outcome and specification."""
    if outcome == Outcome.INFARCTION:
        return data['infarction'], _Family.PROBIT_ORDERED if spec == ModelSpec.SUBMITTED else _Family.LOGIT

    mrs = data[mrs_column]
    if spec == ModelSpec.ORDINAL_LOGIT:
        return mrs, _Family.LOGIT_ORDERED

    # Unfavourable = mRS > 2; submitted counts missing mRS as unfavourable (x <= 2 is False for NaN)
    unfavourable = (~(mrs <= FAVOURABLE_MAX_MRS)).astype(float)
    family = _Family.PROBIT_ORDERED if spec == ModelSpec.SUBMITTED else _Family.LOGIT
    return unfavourable.where(mrs.notna() | (missing == MissingMrs.UNFAVOURABLE)), family


def _favourable_groups(data: pd.DataFrame, mrs_column: str, missing: MissingMrs) -> pd.Series:
    mrs = data[mrs_column]
    favourable = (mrs <= FAVOURABLE_MAX_MRS).astype(float)
    return favourable.where(mrs.notna() | (missing == MissingMrs.UNFAVOURABLE))


def adjusted_effect(data: pd.DataFrame, outcome: Outcome, spec: ModelSpec, missing: MissingMrs,
                    extra_covariates: list[str] = (), mrs_column: str = 'mrs') -> dict:
    """OR per day of DCI onset from haemorrhage, adjusted for COVARIATES + extra_covariates."""
    y, family = _model_frame(data, outcome, spec, missing, mrs_column)
    covariates = [DCI_DAY, *COVARIATES, *extra_covariates]
    frame = pd.concat([y.rename('y'), data[covariates].astype(float)], axis=1).dropna()
    fit = _fit(frame['y'], frame[covariates], family)

    # Events: mRS > 2 or infarction, e.g. 45 of 101
    events = frame['y'] > FAVOURABLE_MAX_MRS if family == _Family.LOGIT_ORDERED else frame['y'] == 1
    return {'outcome': outcome.value, 'model': spec.value, 'n': fit.n, 'n_events': int(events.sum()),
            **_odds_ratio(fit, DCI_DAY)}


def _median_iqr(values: pd.Series) -> str:
    return f'{values.median():.1f} ({values.quantile(0.25):.1f}-{values.quantile(0.75):.1f})'


def unadjusted_comparison(data: pd.DataFrame, outcome: Outcome, missing: MissingMrs, mrs_column: str = 'mrs') -> dict:
    """DCI day by group, Mann-Whitney U; groups mRS <= 2 vs > 2, or infarction vs none."""
    if outcome == Outcome.MRS:
        group = _favourable_groups(data, mrs_column, missing)
        labels = ('mRS > 2', 'mRS <= 2')
    else:
        group = data['infarction']
        labels = ('no infarction', 'infarction')

    days = [data.loc[group == value, DCI_DAY] for value in (0, 1)]
    return {
        'outcome': outcome.value, 'missing_mRS': missing.value if outcome == Outcome.MRS else '',
        'group_0': labels[0], 'n_0': len(days[0]), 'days_0': _median_iqr(days[0]),
        'group_1': labels[1], 'n_1': len(days[1]), 'days_1': _median_iqr(days[1]),
        'p': stats.mannwhitneyu(days[0], days[1]).pvalue,
    }


SPEC_MISSING = {ModelSpec.SUBMITTED: MissingMrs.UNFAVOURABLE, ModelSpec.ORDINAL_LOGIT: MissingMrs.EXCLUDED,
                ModelSpec.BINARY_LOGIT: MissingMrs.EXCLUDED}

# Binary spec differs only for mRS; its infarction model equals the ordinal spec's
SPEC_OUTCOMES = {ModelSpec.SUBMITTED: list(Outcome), ModelSpec.ORDINAL_LOGIT: list(Outcome),
                 ModelSpec.BINARY_LOGIT: [Outcome.MRS]}


def primary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(unadjusted, adjusted) tables for all specifications and both outcomes."""
    unadjusted = [unadjusted_comparison(data, outcome, missing)
                  for outcome in Outcome for missing in (MissingMrs if outcome == Outcome.MRS else [MissingMrs.EXCLUDED])]
    adjusted = [adjusted_effect(data, outcome, spec, SPEC_MISSING[spec]) for spec in ModelSpec for outcome in SPEC_OUTCOMES[spec]]
    return pd.DataFrame(unadjusted), pd.DataFrame(adjusted)


@dataclass(frozen=True)
class GroupComparison:
    """DCI day by binary outcome group with unadjusted and adjusted p, e.g. one Figure 2 panel."""
    outcome: Outcome
    dci_day: pd.Series
    group: pd.Series  # 1 = mRS <= 2 or infarction, 0 = mRS > 2 or none, NaN = excluded
    p_unadjusted: float
    p_adjusted: float


def group_comparison(data: pd.DataFrame, outcome: Outcome, spec: ModelSpec) -> GroupComparison:
    missing = SPEC_MISSING[spec]
    group = _favourable_groups(data, 'mrs', missing) if outcome == Outcome.MRS else data['infarction']
    return GroupComparison(outcome, data[DCI_DAY], group, unadjusted_comparison(data, outcome, missing)['p'],
                           adjusted_effect(data, outcome, spec, missing)['p'])


# --- Sensitivity analyses (ordinal logistic specification), one per reviewer concern ---

@dataclass(frozen=True)
class Sensitivity:
    name: str
    concern: str
    outcomes: list[Outcome]
    select: Callable[[pd.DataFrame], pd.Series] = lambda data: pd.Series(True, index=data.index)
    extra_covariates: list[str] = field(default_factory=list)
    mrs_column: str = 'mrs'


def alive_with_dci_by(day: int) -> Callable[[pd.DataFrame], pd.Series]:
    """Patients alive at `day` with DCI by calendar day `day`, e.g. day 14: DCI on day 14 at 18:00 kept.

    Removes the survival advantage of late DCI: every kept patient could have had DCI at any day up to `day`.
    """
    def select(data: pd.DataFrame) -> pd.Series:
        alive = data['death_day'].isna() | (data['death_day'] > day)
        return alive & (np.floor(data[DCI_DAY]) <= day)
    return select


VISIT_SOURCES = list(FOLLOW_UP_VISITS)
ONE_YEAR_VISIT = VISIT_SOURCES[0]

SENSITIVITY_ANALYSES = [
    Sensitivity('1-year visit or death only (no 2-/5-year substitution)', 'Editor major 4: follow-up window',
                [Outcome.MRS], select=lambda d: d['mrs_source'].isin([ONE_YEAR_VISIT, FOLLOW_UP_DEATH])),
    Sensitivity('Survivors, adjusted for follow-up interval', 'Editor major 4: follow-up window',
                [Outcome.MRS], select=lambda d: d['mrs_source'].isin(VISIT_SOURCES), extra_covariates=['follow_up_months']),
    Sensitivity('Discharge mRS (fixed timepoint)', 'Editor major 4: follow-up window',
                [Outcome.MRS], mrs_column='mrs_discharge'),
    Sensitivity('Adjusted for DCI-related infarction', 'Editor major 1: outcome vs infarction discordance',
                [Outcome.MRS], extra_covariates=['infarction']),
    Sensitivity('In-hospital deaths excluded', 'Editor critical 2: early death / immortal time',
                list(Outcome), select=lambda d: ~d['in_hospital_death']),
    Sensitivity('Adjusted for calendar year', 'Reviewers 1 and 3: change in practice over time',
                list(Outcome), extra_covariates=['year']),
    Sensitivity('Documented ictus date only', 'Reviewer 2: uncertain ictus',
                list(Outcome), select=lambda d: d['ictus_documented']),
    Sensitivity('Clinically assessable at DCI diagnosis', 'Editor critical 1: clinical vs surrogate ascertainment',
                [Outcome.MRS], select=lambda d: d[CLINICAL_ASSESSABILITY] == YES),
    Sensitivity('Missing mRS set to 0 (best case)', 'Editor major 4: missing follow-up',
                [Outcome.MRS], mrs_column='mrs_missing_0'),
    Sensitivity('Missing mRS set to 6 (worst case)', 'Editor major 4: missing follow-up',
                [Outcome.MRS], mrs_column='mrs_missing_6'),
    *[Sensitivity(f'Alive at day {day} with DCI by day {day}', 'Editor critical 2: early death / immortal time',
                  [Outcome.MRS], select=alive_with_dci_by(day)) for day in SURVIVAL_LANDMARK_DAYS],
]


def sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis in SENSITIVITY_ANALYSES:
        subset = data[analysis.select(data)]
        for outcome in analysis.outcomes:
            row = adjusted_effect(subset, outcome, ModelSpec.ORDINAL_LOGIT, MissingMrs.EXCLUDED,
                                  analysis.extra_covariates, analysis.mrs_column)
            rows.append({'analysis': analysis.name, 'concern': analysis.concern, **row})
    return pd.DataFrame(rows)


def proportional_odds_check(data: pd.DataFrame) -> pd.DataFrame:
    """OR per day of the ordinal model vs binary logistic models at each cumulative mRS threshold.

    Proportional odds holds if the binary ORs are similar, e.g. 0.89 (ordinal) vs 0.88 at mRS > 2.
    """
    rows = [{'threshold': 'ordinal (all)', **adjusted_effect(data, Outcome.MRS, ModelSpec.ORDINAL_LOGIT, MissingMrs.EXCLUDED)}]
    covariates = [DCI_DAY, *COVARIATES]
    frame = data[['mrs', *covariates]].astype(float).dropna()

    for threshold in range(MAX_MRS_THRESHOLD + 1):
        y = (frame['mrs'] > threshold).astype(float)
        row = {'threshold': f'mRS > {threshold}', 'outcome': Outcome.MRS.value, 'model': 'binary logistic',
               'n': len(frame), 'n_events': int(y.sum())}
        try:
            row.update(_odds_ratio(_fit(y, frame[covariates], _Family.LOGIT), DCI_DAY))
        except np.linalg.LinAlgError:
            row.update({'OR_per_day': np.nan, 'lower': np.nan, 'upper': np.nan, 'p': np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def linearity_check(data: pd.DataFrame) -> pd.DataFrame:
    """Ordinal mRS model with a quadratic DCI-day term vs the linear model: likelihood-ratio test and AIC."""
    covariates = [DCI_DAY, *COVARIATES]
    frame = data[['mrs', *covariates]].astype(float).dropna()

    # Centred square, e.g. DCI day 12 with mean 9 -> 9; reduces collinearity with the linear term
    frame['dci_day_squared'] = (frame[DCI_DAY] - frame[DCI_DAY].mean()) ** 2
    linear = _fit(frame['mrs'], frame[covariates], _Family.LOGIT_ORDERED)
    quadratic = _fit(frame['mrs'], frame[[*covariates, 'dci_day_squared']], _Family.LOGIT_ORDERED)

    lr = 2 * (quadratic.llf - linear.llf)
    return pd.DataFrame([{'n': linear.n, 'aic_linear': linear.aic, 'aic_quadratic': quadratic.aic, 'lr_statistic': lr,
                          'df': 1, 'p_lr': stats.chi2.sf(lr, 1),
                          'quadratic_coefficient': quadratic.params['dci_day_squared'],
                          'p_quadratic': quadratic.pvalues['dci_day_squared']}])


def follow_up_missingness(data: pd.DataFrame) -> pd.DataFrame:
    """Editor major 4: is missing follow-up mRS associated with DCI timing? Logistic, OR per day."""
    data = data.assign(missing_mrs=data['mrs'].isna().astype(float))
    rows = []
    for label, covariates in [('crude', [DCI_DAY]), ('adjusted', [DCI_DAY, *COVARIATES])]:
        # One model frame per model: the crude model keeps patients with incomplete covariates
        frame = data[['missing_mrs', *covariates]].dropna()
        fit = _fit(frame['missing_mrs'], frame[covariates].astype(float), _Family.LOGIT)
        rows.append({'model': label, 'n': fit.n, 'n_missing': int(frame['missing_mrs'].sum()), **_odds_ratio(fit, DCI_DAY)})
    return pd.DataFrame(rows)


def follow_up_interval(data: pd.DataFrame) -> pd.DataFrame:
    """Follow-up interval (months from admission) and mRS source among DCI patients."""
    months = data['follow_up_months'].dropna()
    sources = data['mrs_source'].fillna('missing').value_counts()
    rows = [{'measure': 'follow-up interval, months, median (IQR)', 'value': _median_iqr(months), 'n': len(months)}]
    rows += [{'measure': f'mRS source: {source}', 'value': '', 'n': int(n)} for source, n in sources.items()]
    return pd.DataFrame(rows)


def severity_association(data: pd.DataFrame) -> pd.DataFrame:
    """Reviewer 2: DCI day vs admission severity. Spearman per grade; linear regression on all covariates."""
    rows = []
    for grade in ['wfns', 'fisher']:
        pair = data[[DCI_DAY, grade]].dropna()
        rho, p = stats.spearmanr(pair[DCI_DAY], pair[grade])
        rows.append({'analysis': f'Spearman, DCI day vs {grade}', 'n': len(pair), 'estimate': rho, 'lower': np.nan,
                     'upper': np.nan, 'p': p})

    frame = data[[DCI_DAY, *COVARIATES]].astype(float).dropna()
    fit = sm.OLS(frame[DCI_DAY], sm.add_constant(frame[COVARIATES])).fit()
    ci = fit.conf_int(alpha=1 - CONFIDENCE)
    for covariate in COVARIATES:
        rows.append({'analysis': f'Linear regression of DCI day (days per unit): {covariate}', 'n': int(fit.nobs),
                     'estimate': fit.params[covariate], 'lower': ci.loc[covariate, 0], 'upper': ci.loc[covariate, 1],
                     'p': fit.pvalues[covariate]})
    return pd.DataFrame(rows)
