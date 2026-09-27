"""One record-linkage rule for every file.

Run from code/: python -m pytest ischemia_timing/late_dci/test_linkage.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import linkage
from ischemia_timing.late_dci.linkage import ID, Match

NAN = np.nan


def _frame(ids, names, births, **columns):
    return pd.DataFrame({ID: ids, 'Name': names, 'Date_birth': pd.to_datetime(births), **columns})


def test_link_follows_the_hierarchy():
    source = _frame(
        ['A', 'B', 'B', 'C', 'C', NAN, NAN, NAN],
        ['a', 'b', 'b', 'c', 'c', 'e', 'f', 'f'],
        ['1960-01-01', '1961-01-01', '1961-01-01', '1962-01-01', '1962-01-01', '1964-01-01', '1965-01-01', '1965-01-01'],
        value=[1, 2, 2, 3, 4, 5, 6, 7],
    )
    # A: ID; B: agreeing duplicates collapse; C: conflicting duplicates; D: ID not in source -> name + birth;
    # no ID -> name + birth; missing name -> no fallback key; F: conflicting name + birth duplicates; unmatched
    target = _frame(
        ['A', 'B', 'C', 'D', NAN, NAN, NAN, 'X'],
        ['a', 'b', 'c', 'e', 'e', NAN, 'f', 'x'],
        ['1960-01-01', '1961-01-01', '1962-01-01', '1964-01-01', '1964-01-01', '1964-01-01', '1965-01-01', '1970-01-01'],
    )

    linked = linkage.link(source, target, ['value'])

    assert linked.values['value'].iloc[[0, 1, 3, 4]].tolist() == [1, 2, 5, 5]
    assert linked.values['value'].iloc[[2, 5, 6, 7]].isna().all()
    assert linked.match.tolist() == [Match.ID, Match.ID, Match.CONFLICT, Match.NAME_BIRTH, Match.NAME_BIRTH,
                                     Match.UNMATCHED, Match.CONFLICT, Match.UNMATCHED]


def test_counts_per_match_type():
    source = _frame(['A'], ['a'], ['1960-01-01'], value=[1])
    target = _frame(['A', NAN, 'Z'], ['a', 'a', 'z'], ['1960-01-01', '1960-01-01', '1960-01-01'])

    counts = linkage.link(source, target, ['value']).counts().set_index('match')['n']

    assert counts.to_dict() == {Match.ID.value: 1, Match.NAME_BIRTH.value: 1, Match.CONFLICT.value: 0,
                                Match.UNMATCHED.value: 1}
