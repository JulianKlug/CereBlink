import numpy as np
import pandas as pd

from utils.metrics import coefficient_of_variation, average_real_variability, complexity_index


def bp_timebin_metrics(timebin_df, normalisation=False, verbose=False) -> pd.DataFrame:
    # Create metrics for negative timebins
    # group by 'negative_timebin' and 'pNr', then obtain median of 'systole', 'diastole', 'mitteldruck
    median_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'median', f'diastole{"_normalised" if normalisation else ""}': 'median', f'mitteldruck{"_normalised" if normalisation else ""}': 'median'}).reset_index()
    median_df = median_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_median', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_median', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_median'})

    max_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'max', f'diastole{"_normalised" if normalisation else ""}': 'max', f'mitteldruck{"_normalised" if normalisation else ""}': 'max'}).reset_index()
    max_df = max_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_max', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_max', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_max'})

    min_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'min', f'diastole{"_normalised" if normalisation else ""}': 'min', f'mitteldruck{"_normalised" if normalisation else ""}': 'min'}).reset_index()
    min_df = min_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_min', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_min', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_min'})

    # Variability measures
    # Coefficient of variation (CV) = standard deviation / mean
    cv_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': coefficient_of_variation, f'diastole{"_normalised" if normalisation else ""}': coefficient_of_variation, f'mitteldruck{"_normalised" if normalisation else ""}': coefficient_of_variation}).reset_index()
    cv_df = cv_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_cv', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_cv', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_cv'})

    # Average real variability (ARV) = sum of absolute differences between consecutive values
    arv_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': average_real_variability, f'diastole{"_normalised" if normalisation else ""}': average_real_variability, f'mitteldruck{"_normalised" if normalisation else ""}': average_real_variability}).reset_index()
    arv_df = arv_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_arv', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_arv', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_arv'})

    # complexity index
    ci_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': complexity_index, f'diastole{"_normalised" if normalisation else ""}': complexity_index, f'mitteldruck{"_normalised" if normalisation else ""}': complexity_index}).reset_index()
    ci_df = ci_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_ci', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_ci', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_ci'})


    # Merge the metrics
    negative_timebin_metrics = pd.merge(median_df, max_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics = pd.merge(negative_timebin_metrics, min_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics = pd.merge(negative_timebin_metrics, cv_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics = pd.merge(negative_timebin_metrics, arv_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics = pd.merge(negative_timebin_metrics, ci_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics['label'] = 0


    # Positive timebins
    pos_median_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'median', f'diastole{"_normalised" if normalisation else ""}': 'median', f'mitteldruck{"_normalised" if normalisation else ""}': 'median'}).reset_index()
    pos_median_df = pos_median_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_median', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_median', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_median'})

    pos_max_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'max', f'diastole{"_normalised" if normalisation else ""}': 'max', f'mitteldruck{"_normalised" if normalisation else ""}': 'max'}).reset_index()
    pos_max_df = pos_max_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_max', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_max', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_max'})

    pos_min_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': 'min', f'diastole{"_normalised" if normalisation else ""}': 'min', f'mitteldruck{"_normalised" if normalisation else ""}': 'min'}).reset_index()
    pos_min_df = pos_min_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_min', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_min', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_min'})

    # Variability measures
    # Coefficient of variation (CV) = standard deviation / mean
    cv_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': coefficient_of_variation, f'diastole{"_normalised" if normalisation else ""}': coefficient_of_variation, f'mitteldruck{"_normalised" if normalisation else ""}': coefficient_of_variation}).reset_index()
    cv_df = cv_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_cv', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_cv', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_cv'})

    # Average real variability (ARV) = sum of absolute differences between consecutive values
    arv_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': average_real_variability, f'diastole{"_normalised" if normalisation else ""}': average_real_variability, f'mitteldruck{"_normalised" if normalisation else ""}': average_real_variability}).reset_index()
    arv_df = arv_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_arv', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_arv', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_arv'})

    # complexity index
    ci_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {f'systole{"_normalised" if normalisation else ""}': complexity_index, f'diastole{"_normalised" if normalisation else ""}': complexity_index, f'mitteldruck{"_normalised" if normalisation else ""}': complexity_index}).reset_index()
    ci_df = ci_df.rename(
        columns={f'systole{"_normalised" if normalisation else ""}': f'systole{"_normalised" if normalisation else ""}_ci', f'diastole{"_normalised" if normalisation else ""}': f'diastole{"_normalised" if normalisation else ""}_ci', f'mitteldruck{"_normalised" if normalisation else ""}': f'mitteldruck{"_normalised" if normalisation else ""}_ci'})

    # Merge the metrics for positive timebins
    pos_timebin_metrics = pd.merge(pos_median_df, pos_max_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics = pd.merge(pos_timebin_metrics, pos_min_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics = pd.merge(pos_timebin_metrics, cv_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics = pd.merge(pos_timebin_metrics, arv_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics = pd.merge(pos_timebin_metrics, ci_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics['label'] = 1

    # Merge the negative and positive timebin metrics
    timebin_metrics_df = pd.concat([negative_timebin_metrics, pos_timebin_metrics])

    n_pos = timebin_metrics_df[timebin_metrics_df['label'] == 1].shape[0]
    n_neg = timebin_metrics_df[timebin_metrics_df['label'] == 0].shape[0]
    log_df = pd.DataFrame({'n_pos': [n_pos], 'n_neg': [n_neg]})

    if verbose:
        print(log_df)

    return timebin_metrics_df, log_df