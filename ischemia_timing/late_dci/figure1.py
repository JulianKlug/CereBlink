"""Figure 1: onset of vasospasm, DCI and DCI-related infarction after haemorrhage, over ICU CT acquisitions.

Days are counted from ictus (admission date if ictus unknown); negative intervals (date errors) are dropped.

    events (left axis)                          CTs (right axis, grey)
    vasospasm  <- Date_CVS_Start (+ fallbacks)  one row per ICU CT, matched to the
    DCI        <- first DCI image date + time   registry by SOS ID + name + birth date
    infarction <- first infarct image date + time
"""
from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_sources import RawSources
from .linkage import link
from .table1 import ID, YearFilter, select_registry

SECONDS_PER_DAY = 86400

# Tails reported in the Results, e.g. 'DCI before day 5 in 5 patients, after day 21 in 3'
EARLY_TAIL_DAY = 5
LATE_TAIL_DAY = 21

VASOSPASM = 'Vasospasm'
DCI = 'DCI'
INFARCTION = 'DCI-related infarct'
CT = 'CT'

# Figure layout as in the submitted manuscript
DAY_RANGE = (0, 30)
N_BINS = 20
BAR_ALPHA = 0.35
FIGURE_SIZE = (5, 5)
COLORS = {CT: 'lightgrey', VASOSPASM: 'turquoise', DCI: 'magenta', INFARCTION: 'blue'}


def _date(value) -> pd.Timestamp:
    # e.g. datetime(2015, 3, 2) or '14.03.2011'; blank -> NaT
    if isinstance(value, (dt.datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(str(value).strip(), dayfirst=True, errors='coerce')


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.map(_date), errors='coerce')


def _time_of_day(value) -> pd.Timedelta:
    # e.g. '12:00', ' 16:26:00', '1900-01-01 18:00:00' or time(18, 0); missing or unparsable -> midnight
    if isinstance(value, dt.time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    if pd.isna(value):
        return pd.Timedelta(0)

    parsed = pd.to_datetime(str(value).strip(), format='mixed', errors='coerce')
    if pd.isna(parsed):
        return pd.Timedelta(0)
    return parsed - parsed.normalize()


def _date_time(dates: pd.Series, times: pd.Series) -> pd.Series:
    # Image date + time of day, e.g. '14.03.2011' + ' 16:26:00'
    return _dates(dates) + pd.to_timedelta(times.map(_time_of_day))


def _days_after(ictus: pd.Series, event: pd.Series) -> pd.Series:
    days = (event - ictus).dt.total_seconds() / SECONDS_PER_DAY
    return days.where(days >= 0)


def _ictus(registry: pd.DataFrame) -> pd.Series:
    return _dates(registry['Date_Ictus']).fillna(_dates(registry['Date_admission']))


def event_days(registry: pd.DataFrame) -> pd.DataFrame:
    """Days from ictus to first vasospasm, DCI and DCI-related infarction; one row per registry patient."""
    ictus = _ictus(registry)
    return pd.DataFrame({
        VASOSPASM: _days_after(ictus, _dates(registry['Date_CVS_Start'])),
        DCI: _days_after(ictus, _date_time(registry['Date_DCI_ischemia_first_image'], registry['Time_DCI_ischemia_first_image'])),
        INFARCTION: _days_after(ictus, _date_time(registry['Date_DCI_infarct_first_image'], registry['Time_DCI_infarct_first_image'])),
    }, index=registry.index)


def ct_days(registry: pd.DataFrame, ct_acquisitions: pd.DataFrame) -> pd.Series:
    """Days from ictus to each ICU CT of the selected registry patients."""
    ictus = link(registry.assign(ictus=_ictus(registry)), ct_acquisitions, ['ictus']).values['ictus']
    return _days_after(pd.to_datetime(ictus), ct_acquisitions['ct_time']).dropna().rename(CT)


def ct_linkage_counts(registry: pd.DataFrame, ct_acquisitions: pd.DataFrame) -> pd.DataFrame:
    """ICU CTs per match type; unmatched includes CTs of patients outside the selected registry rows."""
    return link(registry, ct_acquisitions, [ID]).counts()


def build_figure1_data(sources: RawSources, ct_acquisitions: pd.DataFrame, year_filter: YearFilter,
                       first_year: int) -> tuple[pd.DataFrame, pd.Series]:
    registry = select_registry(sources, year_filter, first_year)
    return event_days(registry), ct_days(registry, ct_acquisitions)


def summarise(events: pd.DataFrame, cts: pd.Series) -> pd.DataFrame:
    """n, median (Q1-Q3), 95th percentile and tail counts (before day 5, after day 21) of days from ictus."""
    rows = []
    for name, days in [*events.items(), (CT, cts)]:
        days = days.dropna()
        rows.append({'distribution': name, 'n': len(days), 'median': days.median(), 'q1': days.quantile(0.25),
                     'q3': days.quantile(0.75), 'p95': days.quantile(0.95),
                     f'n_before_day_{EARLY_TAIL_DAY}': int((days < EARLY_TAIL_DAY).sum()),
                     f'n_after_day_{LATE_TAIL_DAY}': int((days > LATE_TAIL_DAY).sum())})
    return pd.DataFrame(rows)


def plot_figure1(events: pd.DataFrame, cts: pd.Series) -> plt.Figure:
    """Histograms with KDE; events counted on the left axis, CTs on the right axis."""
    sns.set_theme(style='whitegrid')
    fig, ct_axis = plt.subplots(figsize=FIGURE_SIZE)
    event_axis = ct_axis.twinx()

    histogram = dict(alpha=BAR_ALPHA, bins=N_BINS, kde=True, binrange=DAY_RANGE)
    sns.histplot(x=cts.dropna(), color=COLORS[CT], label=CT, ax=ct_axis, **histogram)
    for name, days in events.items():
        sns.histplot(x=days.dropna(), color=COLORS[name], label=name, ax=event_axis, **histogram)

    ct_axis.set_xlabel('Days')
    ct_axis.set_ylabel('Number of CTs')
    event_axis.set_ylabel('Number of events')

    # One legend for both axes
    ct_handles, ct_labels = ct_axis.get_legend_handles_labels()
    event_handles, event_labels = event_axis.get_legend_handles_labels()
    event_axis.legend(ct_handles + event_handles, ct_labels + event_labels)

    # CT axis on the right, event axis on the left; horizontal grid follows the event axis
    ct_axis.yaxis.grid(False)
    ct_axis.yaxis.tick_right()
    ct_axis.yaxis.set_label_position('right')
    event_axis.yaxis.tick_left()
    event_axis.yaxis.set_label_position('left')

    ct_axis.set_xlim(0, DAY_RANGE[1])
    return fig
