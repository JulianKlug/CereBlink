"""Patient table and analysis datasets for the late-onset DCI analysis.

All times are in days from ictus (day 0). Example: DCI imaged on day 9 at 12:00
-> t_dci = 9.5.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

import numpy as np
import pandas as pd

from .data_sources import RawSources

ID = 'SOS-CENTER-YEAR-NO.'
MISSING_MARKER = 'none'
SECONDS_PER_DAY = 86400
DAYS_PER_YEAR = 365.25

DEAD_MRS = 6
POOR_WFNS_MIN = 4
ACTIVE_SMOKER_CODE = 1
MALE_CODE = 'm'

LANDMARK_DAY = 7.0
FOLLOW_UP_CAP_DAY = 21.0

PCT_COUNT_COLUMN = 'Number of perfusion CTs'
# Perfusion findings recorded at DCI diagnosis; 'na' = not available, like a blank cell
PERFUSION_METRICS = ['TTP_increase', 'TTD_increase', 'Tmax_increase', 'MTT_increased', 'CBV_reduced', 'CBF_reduced']
NOT_AVAILABLE_MARKER = 'na'

CORE_COVARIATES = ['age', 'male', 'hypertension', 'poor_wfns', 'fisher', 'active_smoker', 'aspirin', 'year']
EXTRA_COVARIATES = ['alcohol', 'diabetes', 'statin', 'clopidogrel', 'oral_anticoagulation']
EXTENDED_COVARIATES = CORE_COVARIATES + EXTRA_COVARIATES

# registry column -> analysis covariate, for 0/1 coded variables
BINARY_REGISTRY_COLUMNS = {
    'HTN': 'hypertension',
    'ASS': 'aspirin',
    'Drinker': 'alcohol',
    'DM': 'diabetes',
    'Statin': 'statin',
    'Clopidogrel': 'clopidogrel',
    'OAC': 'oral_anticoagulation',
}


class Event(IntEnum):
    CENSORED = 0
    DCI = 1
    DEATH = 2


class FollowUpEnd(Enum):
    HOSPITAL_DISCHARGE = 'hospital discharge'
    ICU_DISCHARGE = 'ICU discharge'


class AnalysisSet(Enum):
    FULL_COHORT = 'full cohort'
    COMPLETE_CASE = 'complete case'


@dataclass(frozen=True)
class Selection:
    patients: pd.DataFrame
    flow: pd.DataFrame


@dataclass(frozen=True)
class LandmarkData:
    dataset: pd.DataFrame
    flow: pd.DataFrame


def _to_date(value) -> pd.Timestamp:
    if pd.isna(value) or value == MISSING_MARKER:
        return pd.NaT

    if isinstance(value, (dt.datetime, pd.Timestamp)):
        return pd.Timestamp(value)

    # e.g. '16.05.2013' or '2019-08-28 00:00:00'
    return pd.to_datetime(str(value).strip(), dayfirst=True, errors='coerce')


def _to_time_of_day(value) -> pd.Timedelta:
    # Missing time of day -> midnight, e.g. 'none' -> 0h
    if pd.isna(value) or value == MISSING_MARKER:
        return pd.Timedelta(0)

    if isinstance(value, dt.time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute)

    parsed = pd.to_datetime(str(value).strip(), format='mixed', errors='coerce')
    if pd.isna(parsed):
        return pd.Timedelta(0)
    return pd.Timedelta(hours=parsed.hour, minutes=parsed.minute)


def _dates(series: pd.Series) -> pd.Series:
    # Out-of-range typos (e.g. year 3026) -> NaT
    return pd.to_datetime(series.map(_to_date), errors='coerce')


def _days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / SECONDS_PER_DAY


def _name_birth_key(frame: pd.DataFrame) -> pd.Series:
    # Fallback join key for rows without an SOS ID, e.g. 'jane doe|1960-01-31'
    name = frame['Name'].astype(str).str.strip().str.lower()
    birth = pd.to_datetime(frame['Date_birth'], errors='coerce').dt.date.astype(str)
    return name + '|' + birth


def _lookup(source: pd.DataFrame, column: str, timings: pd.DataFrame) -> pd.Series:
    """Value of `column` in `source` for every timings row: by SOS ID, else by name + birth date."""
    by_id = source.dropna(subset=[ID]).drop_duplicates(ID).set_index(ID)[column]
    by_key = source.assign(key=_name_birth_key(source)).drop_duplicates('key').set_index('key')[column]

    matched_by_id = timings[ID].map(by_id)
    matched_by_key = _name_birth_key(timings).map(by_key)
    return matched_by_id.where(timings[ID].notna(), matched_by_key)


def build_patients(sources: RawSources) -> pd.DataFrame:
    """One row per patient of the DCI timings file, with harmonised times and covariates."""
    timings = sources.dci_timings.reset_index(drop=True)
    registry = sources.registry
    outcomes = sources.outcomes

    def reg(column):
        return _lookup(registry, column, timings)

    def out(column):
        return _lookup(outcomes, column, timings)

    patients = pd.DataFrame(index=timings.index)

    # Ictus: timings file, else outcomes file; registry wins on disagreement (typos, e.g. 2003 for 2013)
    ictus_timings = _dates(timings['Date_Ictus'])
    ictus_registry = _dates(reg('Date_Ictus'))
    registry_disagrees = ictus_timings.notna() & ictus_registry.notna() & (ictus_timings != ictus_registry)
    ictus = ictus_timings.where(~registry_disagrees, ictus_registry).fillna(_dates(out('Date_Ictus')))
    patients['ictus_source'] = np.where(registry_disagrees, 'registry',
                                        np.where(ictus_timings.notna(), 'timings', np.where(ictus.notna(), 'outcomes', 'missing')))

    # DCI: verified status, onset = first DCI image date + time
    dci_onset = _dates(timings['Date_DCI_ischemia_first_image']) + timings['Time_DCI_ischemia_first_image'].map(_to_time_of_day)
    patients['dci_status_known'] = timings['DCI_YN_verified'].notna()
    patients['dci'] = timings['DCI_YN_verified'] == 1
    patients['t_dci'] = _days_between(ictus, dci_onset).where(patients['dci'])

    # Hospital discharge: timings, else registry, else ICU discharge as lower bound
    icu_discharge = _dates(reg('Date_discharge_ICU')).fillna(_dates(out('Date_discharge_ICU')))
    discharge = pd.to_datetime(timings['Date_Discharge']).fillna(_dates(reg('Date_Discharge'))).fillna(icu_discharge)
    patients['t_discharge'] = _days_between(ictus, discharge)
    patients['t_icu_discharge'] = _days_between(ictus, icu_discharge)

    # Death: registry, else in-hospital death from outcomes (discharge mRS 6)
    death_registry = pd.to_numeric(reg('Death'), errors='coerce')
    discharge_mrs = pd.to_numeric(out('mRS_discharge'), errors='coerce')
    death_outcomes = (discharge_mrs == DEAD_MRS).astype(float).where(discharge_mrs.notna())
    patients['death'] = death_registry.fillna(death_outcomes)
    patients['death_source'] = np.where(death_registry.notna(), 'registry', np.where(death_outcomes.notna(), 'outcomes', 'missing'))

    # Death date: registry, else hospital discharge date
    death_date = _dates(reg('Date_Death')).fillna(discharge.where(patients['death'] == 1))
    patients['t_death'] = _days_between(ictus, death_date).where(patients['death'] == 1)

    # Covariates (registry)
    age = pd.to_numeric(reg('Age'), errors='coerce')
    age_from_birth = _days_between(pd.to_datetime(timings['Date_birth']), ictus) / DAYS_PER_YEAR
    patients['age_recomputed'] = age < 0
    patients['age'] = age.where(age >= 0, age_from_birth)

    sex = reg('Sex').astype(str).str.strip().str.lower().where(reg('Sex').notna())
    patients['male'] = (sex == MALE_CODE).astype(float).where(sex.notna())

    wfns = pd.to_numeric(reg('WFNS'), errors='coerce')
    patients['poor_wfns'] = (wfns >= POOR_WFNS_MIN).astype(float).where(wfns.notna())
    patients['fisher'] = pd.to_numeric(reg('Fisher_Score'), errors='coerce')

    smoking = pd.to_numeric(reg('Smoker_0no_1yes_2ex'), errors='coerce')
    patients['active_smoker'] = (smoking == ACTIVE_SMOKER_CODE).astype(float).where(smoking.notna())

    for registry_column, covariate in BINARY_REGISTRY_COLUMNS.items():
        patients[covariate] = pd.to_numeric(reg(registry_column), errors='coerce')

    patients['year'] = ictus.dt.year.astype(float)

    # Imaging use: perfusion CTs per patient, and whether each perfusion metric was reported
    patients['n_pct'] = pd.to_numeric(_lookup(sources.pct_counts, PCT_COUNT_COLUMN, timings), errors='coerce')
    for metric in PERFUSION_METRICS:
        patients[f'{metric}_available'] = timings[metric].notna() & (timings[metric] != NOT_AVAILABLE_MARKER)

    return patients


def select(patients: pd.DataFrame, analysis_set: AnalysisSet, covariates: list[str] = CORE_COVARIATES) -> Selection:
    """Apply inclusion criteria in order and record the exclusion flow."""
    death_known = patients['death'].notna() & ((patients['death'] == 0) | patients['t_death'].notna())
    dci_onset_valid = ~patients['dci'] | (patients['t_dci'] >= 0)

    criteria = [
        ('DCI status known', patients['dci_status_known']),
        ('ictus date known', patients['ictus_source'] != 'missing'),
        ('DCI onset date valid', dci_onset_valid),
        ('death status known', death_known),
        ('discharge date known', patients['t_discharge'].notna()),
    ]
    if analysis_set == AnalysisSet.COMPLETE_CASE:
        criteria.append(('covariates complete', patients[covariates].notna().all(axis=1)))

    keep = pd.Series(True, index=patients.index)
    flow = [('timings file rows', len(patients))]
    for label, criterion in criteria:
        keep &= criterion
        flow.append((label, int(keep.sum())))

    return Selection(patients=patients[keep].copy(), flow=pd.DataFrame(flow, columns=['step', 'n']))


def _dci_before_death(patients: pd.DataFrame) -> pd.Series:
    no_death = patients['death'] != 1
    return patients['dci'] & (no_death | (patients['t_dci'] <= patients['t_death']))


def build_landmark_dataset(
    patients: pd.DataFrame,
    landmark_day: float = LANDMARK_DAY,
    cap_day: Optional[float] = FOLLOW_UP_CAP_DAY,
    follow_up_end: FollowUpEnd = FollowUpEnd.HOSPITAL_DISCHARGE,
) -> LandmarkData:
    """Patients alive, DCI-free and observed at the landmark; time counted from the landmark.

    Example (landmark 7, cap 21): DCI on day 9.5 -> time 2.5, event DCI;
    discharged alive on day 30 -> time 14, censored.
    """
    end_column = 't_discharge' if follow_up_end == FollowUpEnd.HOSPITAL_DISCHARGE else 't_icu_discharge'
    follow_up_end_day = patients[end_column]
    observation_end = np.minimum(follow_up_end_day, np.inf if cap_day is None else cap_day)
    dci_first = _dci_before_death(patients)
    is_dead = patients['death'] == 1

    # Exits before the landmark, in priority order
    end_unknown = follow_up_end_day.isna()
    early_dci = ~end_unknown & dci_first & (patients['t_dci'] <= landmark_day)
    early_death = ~end_unknown & ~early_dci & is_dead & (patients['t_death'] <= landmark_day)
    early_exit = ~end_unknown & ~early_dci & ~early_death & (follow_up_end_day <= landmark_day)
    at_risk = ~(end_unknown | early_dci | early_death | early_exit)

    # First event within (landmark, observation end]
    dci_event = dci_first & (patients['t_dci'] <= observation_end)
    death_event = ~dci_event & is_dead & (patients['t_death'] <= observation_end)
    exit_day = np.where(dci_event, patients['t_dci'], np.where(death_event, patients['t_death'], observation_end))
    event = np.where(dci_event, Event.DCI, np.where(death_event, Event.DEATH, Event.CENSORED))

    dataset = patients.assign(time=exit_day - landmark_day, event=event.astype(int))[at_risk].copy()

    flow = pd.DataFrame([
        ('selected', len(patients)),
        (f'{follow_up_end.value} date unknown', int(end_unknown.sum())),
        (f'DCI <= day {landmark_day:g}', int(early_dci.sum())),
        (f'death without DCI <= day {landmark_day:g}', int(early_death.sum())),
        (f'{follow_up_end.value} alive <= day {landmark_day:g}', int(early_exit.sum())),
        (f'risk set at day {landmark_day:g}', len(dataset)),
        ('late DCI', int((dataset['event'] == Event.DCI).sum())),
        ('competing death', int((dataset['event'] == Event.DEATH).sum())),
        ('censored', int((dataset['event'] == Event.CENSORED).sum())),
    ], columns=['step', 'n'])

    return LandmarkData(dataset=dataset, flow=flow)


def build_piecewise_dataset(
    patients: pd.DataFrame,
    split_day: float = LANDMARK_DAY,
    cap_day: float = FOLLOW_UP_CAP_DAY,
) -> pd.DataFrame:
    """Counting-process rows from ictus, split at `split_day`; death censors (cause-specific).

    Example (split 7): DCI on day 9.5 -> rows (0, 7] no event and (7, 9.5] event, late_period = 0 / 1.
    """
    observation_end = np.minimum(patients['t_discharge'], cap_day)
    dci_event = _dci_before_death(patients) & (patients['t_dci'] <= observation_end)
    death_first = ~dci_event & (patients['death'] == 1) & (patients['t_death'] <= observation_end)
    exit_day = np.where(dci_event, patients['t_dci'], np.where(death_first, patients['t_death'], observation_end))

    base = patients.assign(exit_day=exit_day, dci_event=dci_event.astype(int))
    base = base[base['exit_day'] > 0].reset_index().rename(columns={'index': 'patient'})

    early = base.assign(
        start=0.0,
        stop=np.minimum(base['exit_day'], split_day),
        event=np.where(base['exit_day'] <= split_day, base['dci_event'], 0),
        late_period=0,
    )
    late = base[base['exit_day'] > split_day].assign(start=split_day, event=lambda d: d['dci_event'], late_period=1)
    late['stop'] = late['exit_day']

    rows = pd.concat([early, late], ignore_index=True)
    return rows.drop(columns=['exit_day', 'dci_event']).sort_values(['patient', 'start']).reset_index(drop=True)
