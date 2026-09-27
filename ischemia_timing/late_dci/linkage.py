"""Record linkage between the registry, outcomes, DCI timings, pCT counts and CT files.

One rule for every file, per target row:

    SOS ID found in source ──────────────▶ ID match
      └─ else name + birth date found ───▶ name + birth match   (key only if both present)
    duplicate source keys: identical values collapse, differing values -> conflict (NaN)
    nothing found ───────────────────────▶ unmatched (NaN)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

ID = 'SOS-CENTER-YEAR-NO.'
_KEY = '_key'


class Match(Enum):
    ID = 'SOS ID'
    NAME_BIRTH = 'name + birth date'
    CONFLICT = 'duplicate conflict'
    UNMATCHED = 'unmatched'


@dataclass(frozen=True)
class Linked:
    values: pd.DataFrame  # source columns, one row per target row
    match: pd.Series      # Match per target row

    def counts(self) -> pd.DataFrame:
        """Target rows per match type, e.g. SOS ID 450, name + birth date 20, conflict 1, unmatched 10."""
        return pd.DataFrame({'match': [match.value for match in Match],
                             'n': [int((self.match == match).sum()) for match in Match]})


def _name_birth_key(frame: pd.DataFrame) -> pd.Series:
    # e.g. 'jane doe|1960-01-31'; missing name or birth date -> no key
    name = frame['Name'].where(frame['Name'].notna()).astype(str).str.strip().str.lower()
    birth = pd.to_datetime(frame['Date_birth'], errors='coerce')
    key = name + '|' + birth.dt.date.astype(str)
    return key.where(frame['Name'].notna() & (name != '') & birth.notna())


def _id_key(frame: pd.DataFrame) -> pd.Series:
    return frame[ID].where(frame[ID].notna())


def _unique_and_conflicting(source: pd.DataFrame, key: pd.Series, columns: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    # Identical duplicates collapse; keys with differing values are conflicts
    keyed = source[columns].assign(**{_KEY: key}).dropna(subset=[_KEY]).drop_duplicates()
    duplicated = keyed[_KEY].duplicated(keep=False)
    unique = keyed[~duplicated].set_index(_KEY)[columns]
    return unique, pd.Index(keyed.loc[duplicated, _KEY].unique())


def link(source: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> Linked:
    """`columns` of `source` for every `target` row, following the hierarchy of the module docstring."""
    values = pd.DataFrame(index=target.index, columns=columns, dtype=object)
    match = pd.Series(Match.UNMATCHED, index=target.index, dtype=object)
    resolved = pd.Series(False, index=target.index)

    for key_function, match_type in [(_id_key, Match.ID), (_name_birth_key, Match.NAME_BIRTH)]:
        unique, conflicts = _unique_and_conflicting(source, key_function(source), columns)
        target_key = key_function(target)

        found = ~resolved & target_key.isin(unique.index)
        values.loc[found, columns] = unique.reindex(target_key[found]).to_numpy()
        match[found] = match_type

        conflicting = ~resolved & ~found & target_key.isin(conflicts)
        match[conflicting] = Match.CONFLICT
        resolved |= found | conflicting

    return Linked(values=values.infer_objects(), match=match)
