"""Table 1: registry population overall and by DCI.

Cells are 'n (%)' or 'median (Q1-Q3)'; percentages use patients with a known value.
Missing values are appended, e.g. '33 (8.5%) [19 missing]'.

Outcome data (mRS) are matched to registry rows by SOS ID, else by name + birth date.
Outcome rows sharing an SOS ID conflict (e.g. discharge mRS 6 vs 2) and count as missing.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

import pandas as pd

from utils.utils import safe_conversion_to_datetime

from .data_sources import RawSources

ID = 'SOS-CENTER-YEAR-NO.'
DEAD_MRS = 6
DAYS_PER_YEAR = 365.25
MALE_CODES = ['M', 'm']
YES = 1
NO = 0
ACTIVE_SMOKER_CODE = 1
FISHER_MISSING_MARKERS = {'x': pd.NA, 'nan': pd.NA}
OUTCOME_COLUMNS = ['mRS_discharge', 'mRS_FU_1y']

# Visit behind mRS_FU_1y, e.g. '2y' when the 1-year score is missing; 'death' when set to 6
FOLLOW_UP_VISITS = {'1y': ('mRS_FU_1y', 'Date_FU_1y'), '2y': ('mRS_2FU_2y', 'Date_2FU_2y'), '5y': ('mRS_3FU_5y', 'Date_3FU_5y')}
FOLLOW_UP_DEATH = 'death'
FOLLOW_UP_COLUMNS = ['mRS_FU_source', 'Date_FU_used']
LINKED_COLUMNS = OUTCOME_COLUMNS + FOLLOW_UP_COLUMNS

# DCI status and image dates / times, taken from the verified DCI timings file instead of the registry
DCI_DATE_COLUMNS = ['Date_DCI_ischemia_first_image', 'Time_DCI_ischemia_first_image',
                    'Date_DCI_infarct_first_image', 'Time_DCI_infarct_first_image']
DCI_VERIFIED = 'DCI_YN_verified'

# Earlier CVS detection dates fill a missing start date, in this order
CVS_DATE_FALLBACKS = ['Date_CVS_DSA', 'Date_CVS_CTA', 'Date_CVS_TCD']

# Registry artery code -> Table 1 location
ANEURYSM_LOCATIONS = {
    'Anterior communicating artery': [8],
    'Anterior cerebral artery': [9, 22, 24],
    'Middle cerebral artery': [7, 20, 21],
    'Posterior cerebral artery': [23, 28],
    'Internal carotid artery': [1, 2, 3, 4, 5, 6, 18, 19, 25, 26, 27, 29, 31],
    'Vertebral/basilar artery': [10, 11, 12, 13, 14, 15, 16, 17],
}
MAPPED_ANEURYSM_CODES = {code for codes in ANEURYSM_LOCATIONS.values() for code in codes}

GROUP_ALL = 'Overall Population'
GROUP_DCI = 'DCI'
GROUP_NO_DCI = 'No DCI'


class YearFilter(Enum):
    STUDY_PERIOD = 'study period'
    ALL_YEARS = 'all years'


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.apply(safe_conversion_to_datetime), errors='coerce')


def _name_birth_key(frame: pd.DataFrame) -> pd.Series:
    # Fallback join key for rows without an SOS ID, e.g. 'jane doe|1960-01-31'
    name = frame['Name'].astype(str).str.strip().str.lower()
    birth = pd.to_datetime(frame['Date_birth'], errors='coerce').dt.date.astype(str)
    return name + '|' + birth


def _prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    outcomes = outcomes.copy()

    # Earliest visit with a score wins, as in the fill below; raw date kept, e.g. '14/08/2023'
    outcomes[FOLLOW_UP_COLUMNS] = pd.NA
    for source, (score, date) in reversed(FOLLOW_UP_VISITS.items()):
        scored = outcomes[score].notna()
        outcomes.loc[scored, 'mRS_FU_source'] = source
        outcomes.loc[scored, 'Date_FU_used'] = outcomes.loc[scored, date]

    # 1-year mRS: else 2-year, else 5-year; in-hospital death -> 6
    outcomes['mRS_FU_1y'] = outcomes['mRS_FU_1y'].fillna(outcomes['mRS_2FU_2y']).fillna(outcomes['mRS_3FU_5y'])
    outcomes.loc[outcomes['mRS_discharge'] == DEAD_MRS, 'mRS_FU_1y'] = DEAD_MRS
    outcomes.loc[outcomes['mRS_discharge'] == DEAD_MRS, FOLLOW_UP_COLUMNS] = [FOLLOW_UP_DEATH, pd.NA]

    for column in OUTCOME_COLUMNS:
        outcomes[column] = pd.to_numeric(outcomes[column], errors='coerce')
    return outcomes


def _outcome_lookup(outcomes: pd.DataFrame, registry: pd.DataFrame, columns: list[str] = LINKED_COLUMNS) -> pd.DataFrame:
    """`columns` of `outcomes` for every registry row: by SOS ID, else by name + birth date; duplicate keys -> NaN."""
    def unique_by(key: pd.Series) -> pd.DataFrame:
        keyed = outcomes[columns].set_index(key)
        keyed = keyed[keyed.index.notna()]
        return keyed[~keyed.index.duplicated(keep=False)]

    by_id = unique_by(outcomes[ID])
    by_key = unique_by(_name_birth_key(outcomes))

    matched_by_id = by_id.reindex(registry[ID]).set_axis(registry.index)
    matched_by_key = by_key.reindex(_name_birth_key(registry)).set_axis(registry.index)
    return matched_by_id.where(registry[ID].notna(), matched_by_key)


def timings_dci_dates(timings: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """DCI image dates / times of the timings file for every registry row; blank unless DCI verified.

    e.g. registry DCI on 01.01.2015, timings (verified) 05.01.2015 -> 05.01.2015; not verified -> blank.
    """
    verified = timings.assign(**{column: timings[column].where(timings[DCI_VERIFIED] == YES) for column in DCI_DATE_COLUMNS})
    return _outcome_lookup(verified, registry, DCI_DATE_COLUMNS)


def timings_dci_status(timings: pd.DataFrame, registry: pd.DataFrame) -> pd.Series:
    """Verified DCI status (1 / 0) of the timings file for every registry row; absent from timings -> NaN."""
    return _outcome_lookup(timings, registry, [DCI_VERIFIED])[DCI_VERIFIED]


def _age(registry: pd.DataFrame) -> pd.Series:
    # Negative registry age = sign error, e.g. -45 -> recomputed from birth date to ictus (admission if unknown)
    ictus = _dates(registry['Date_Ictus']).fillna(_dates(registry['Date_admission']))
    from_birth = (ictus - pd.to_datetime(registry['Date_birth'], errors='coerce')).dt.days / DAYS_PER_YEAR
    age = pd.to_numeric(registry['Age'], errors='coerce')
    return age.where(age >= 0, from_birth)


def _prepare_registry(registry: pd.DataFrame, outcomes: pd.DataFrame, timings: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()
    registry[DCI_DATE_COLUMNS] = timings_dci_dates(timings, registry)
    registry['DCI_ischemia'] = timings_dci_status(timings, registry)
    registry['Age'] = _age(registry)

    # CVS start: first available detection date; a dated CVS implies CVS_YN = 1
    for fallback in CVS_DATE_FALLBACKS:
        missing = registry['Date_CVS_Start'].isnull() & registry[fallback].notnull()
        registry.loc[missing, 'Date_CVS_Start'] = registry[fallback]
    dated_cvs = registry['Date_CVS_Start'].apply(safe_conversion_to_datetime).notnull()
    registry.loc[(registry['CVS_YN'] == 0) & dated_cvs, 'CVS_YN'] = 1

    registry['Fisher_Score'] = pd.to_numeric(registry['Fisher_Score'].replace(FISHER_MISSING_MARKERS), errors='coerce')
    registry['male'] = registry['Sex'].isin(MALE_CODES).astype(float).where(registry['Sex'].notna())

    # Endovascular treatment: coiling or stenting; unknown only if neither is known to be done
    endovascular = (registry['Coiling'] == YES) | (registry['Stenting'] == YES)
    unknown = ~endovascular & (registry['Coiling'].isna() | registry['Stenting'].isna())
    registry['coiling'] = endovascular.astype(float).where(~unknown)

    # First artery code only, e.g. '7, 20' -> 7, '8 (left)' -> 8
    code = registry['Aneurysm_Artery_Code'].astype(str).str.split(',').str[0].str.split('(').str[0]
    registry['aneurysm_code'] = pd.to_numeric(code.replace({'nan': pd.NA, 'NaN': pd.NA, 'NA': pd.NA}), errors='raise')

    admission = _dates(registry['Date_admission'])
    registry['los'] = (_dates(registry['Date_Discharge']) - admission).dt.days
    registry['los_icu'] = (_dates(registry['Date_discharge_ICU']) - admission).dt.days

    registry[LINKED_COLUMNS] = _outcome_lookup(outcomes, registry)

    # Death after discharge (registry) -> 1-year mRS 6
    registry.loc[registry['Death'] == YES, 'mRS_FU_1y'] = DEAD_MRS
    registry.loc[registry['Death'] == YES, FOLLOW_UP_COLUMNS] = [FOLLOW_UP_DEATH, pd.NA]
    return registry


def select_registry(sources: RawSources, year_filter: YearFilter, first_year: int) -> pd.DataFrame:
    """Registry rows with a known admission date, optionally from `first_year` on, with outcomes attached."""
    admission = sources.registry['Date_admission']
    keep = admission.notna()
    if year_filter == YearFilter.STUDY_PERIOD:
        keep &= admission >= f'{first_year}-01-01'
    return _prepare_registry(sources.registry[keep], _prepare_outcomes(sources.outcomes), sources.dci_timings)


def population_flow(registry: pd.DataFrame, timings: pd.DataFrame, year_filter: YearFilter, first_year: int) -> pd.DataFrame:
    """Patients remaining after each selection step of Table 1, then split by verified DCI status.

    e.g. 488 registry rows -> 460 with admission date -> 408 from 2011 -> 392 with verified status (109 DCI, 283 no DCI).
    """
    admission = registry['Date_admission']
    keep = admission.notna()
    steps = [('registry rows', len(registry)), ('admission date known', int(keep.sum()))]

    if year_filter == YearFilter.STUDY_PERIOD:
        keep &= admission >= f'{first_year}-01-01'
        steps.append((f'admission from {first_year}', int(keep.sum())))

    status = timings_dci_status(timings, registry[keep])
    steps += [('verified DCI status known', int(status.notna().sum())),
              ('  DCI', int((status == YES).sum())), ('  no DCI', int((status == NO).sum()))]
    return pd.DataFrame(steps, columns=['step', 'n'])


def _with_missing(text: str, n_missing: int) -> str:
    return f'{text} [{n_missing} missing]' if n_missing else text


def _count(column: str, value=YES) -> Callable[[pd.DataFrame], str]:
    # e.g. aspirin: 33 of 389 known -> '33 (8.5%) [19 missing]'
    def cell(group: pd.DataFrame) -> str:
        values = group[column]
        known = values.notna().sum()
        n = (values == value).sum()
        return _with_missing(f'{n} ({n / known * 100:.1f}%)', values.isna().sum())
    return cell


def _median_iqr(column: str, decimals: int = 0) -> Callable[[pd.DataFrame], str]:
    def cell(group: pd.DataFrame) -> str:
        values = group[column]
        text = f'{values.median():.{decimals}f} ({values.quantile(0.25):.{decimals}f}-{values.quantile(0.75):.{decimals}f})'
        return _with_missing(text, values.isna().sum())
    return cell


def _location(codes: Optional[list[int]]) -> Callable[[pd.DataFrame], str]:
    # Percent of patients with a known location; codes None -> codes outside the mapped locations
    def cell(group: pd.DataFrame) -> str:
        code = group['aneurysm_code'].dropna()
        is_location = code.isin(codes) if codes else ~code.isin(MAPPED_ANEURYSM_CODES)
        return f'{is_location.sum()} ({is_location.mean() * 100:.1f}%)'
    return cell


def _location_missing(group: pd.DataFrame) -> str:
    return str(group['aneurysm_code'].isna().sum())


# (label, cell function); cell function None -> section header
ROWS: list[tuple[str, Optional[Callable[[pd.DataFrame], str]]]] = [
    ('Demographics', None),
    ('  Age', _median_iqr('Age', decimals=1)),
    ('  Sex (male)', _count('male')),
    ('Risk factors', None),
    ('  Hypertension', _count('HTN')),
    ('  Diabetes mellitus', _count('DM')),
    ('  Smoking (active)', _count('Smoker_0no_1yes_2ex', ACTIVE_SMOKER_CODE)),
    ('  Alcohol abuse', _count('Drinker')),
    ('Medication history', None),
    ('  Statin', _count('Statin')),
    ('  Acetylsalicylic acid', _count('ASS')),
    ('  Clopidogrel', _count('Clopidogrel')),
    ('  Oral anticoagulation', _count('OAC')),
    ('Aneurysm location', None),
    *[(f'  {location}', _location(codes)) for location, codes in ANEURYSM_LOCATIONS.items()],
    ('  Other locations', _location(None)),
    ('  Missing', _location_missing),
    ('Admission status', None),
    ('  Glasgow Coma Scale', _median_iqr('GCS_admission')),
    ('  World Federation of Neurological Surgeons Scale', _median_iqr('WFNS')),
    ('  Modified Fisher Scale', _median_iqr('Fisher_Score')),
    ('Acute treatment', None),
    ('  Coiling', _count('coiling')),
    ('  Clipping', _count('Clipping')),
    ('Outcomes', None),
    ('  Vasospasm', _count('CVS_YN')),
    ('  DCI related infarction', _count('DCI_infarct')),
    ('  ICU length of stay (d)', _median_iqr('los_icu')),
    ('  Hospital length of stay (d)', _median_iqr('los')),
    ('  Hospital mortality', _count('Death')),
    ('  Discharge modified Rankin Scale', _median_iqr('mRS_discharge')),
    ('  1-yr modified Rankin Scale', _median_iqr('mRS_FU_1y')),
]


def build_table1(sources: RawSources, year_filter: YearFilter, first_year: int) -> pd.DataFrame:
    """Formatted Table 1; columns 'Overall Population\\n(n = 408)', 'DCI\\n(n = ...)', 'No DCI\\n(n = ...)'."""
    registry = select_registry(sources, year_filter, first_year)
    groups = {
        GROUP_ALL: registry,
        GROUP_DCI: registry[registry['DCI_ischemia'] == 1],
        GROUP_NO_DCI: registry[registry['DCI_ischemia'] == 0],
    }

    columns = {'Variable': [label for label, _ in ROWS]}
    for name, group in groups.items():
        columns[f'{name}\n(n = {len(group)})'] = ['' if cell is None else cell(group) for _, cell in ROWS]
    return pd.DataFrame(columns)
