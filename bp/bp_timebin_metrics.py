import pandas as pd


def bp_timebin_metrics(timebin_df, verbose=False) -> pd.DataFrame:
    # Create metrics for negative timebins
    # group by 'negative_timebin' and 'pNr', then obtain median of 'systole', 'diastole', 'mitteldruck'
    median_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {'systole': 'median', 'diastole': 'median', 'mitteldruck': 'median'}).reset_index()
    median_df = median_df.rename(
        columns={'systole': 'systole_median', 'diastole': 'diastole_median', 'mitteldruck': 'mitteldruck_median'})

    max_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {'systole': 'max', 'diastole': 'max', 'mitteldruck': 'max'}).reset_index()
    max_df = max_df.rename(
        columns={'systole': 'systole_max', 'diastole': 'diastole_max', 'mitteldruck': 'mitteldruck_max'})

    min_df = timebin_df.groupby(['negative_timebin', 'pNr']).agg(
        {'systole': 'min', 'diastole': 'min', 'mitteldruck': 'min'}).reset_index()
    min_df = min_df.rename(
        columns={'systole': 'systole_min', 'diastole': 'diastole_min', 'mitteldruck': 'mitteldruck_min'})

    # Merge the metrics
    negative_timebin_metrics = pd.merge(median_df, max_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics = pd.merge(negative_timebin_metrics, min_df, on=['negative_timebin', 'pNr'])
    negative_timebin_metrics['label'] = 0


    # Positive timebins
    pos_median_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {'systole': 'median', 'diastole': 'median', 'mitteldruck': 'median'}).reset_index()
    pos_median_df = pos_median_df.rename(
        columns={'systole': 'systole_median', 'diastole': 'diastole_median', 'mitteldruck': 'mitteldruck_median'})

    pos_max_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {'systole': 'max', 'diastole': 'max', 'mitteldruck': 'max'}).reset_index()
    pos_max_df = pos_max_df.rename(
        columns={'systole': 'systole_max', 'diastole': 'diastole_max', 'mitteldruck': 'mitteldruck_max'})

    pos_min_df = timebin_df.groupby(['associated_event_time', 'pNr']).agg(
        {'systole': 'min', 'diastole': 'min', 'mitteldruck': 'min'}).reset_index()
    pos_min_df = pos_min_df.rename(
        columns={'systole': 'systole_min', 'diastole': 'diastole_min', 'mitteldruck': 'mitteldruck_min'})

    # Merge the metrics for positive timebins
    pos_timebin_metrics = pd.merge(pos_median_df, pos_max_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics = pd.merge(pos_timebin_metrics, pos_min_df, on=['associated_event_time', 'pNr'])
    pos_timebin_metrics['label'] = 1

    # Merge the negative and positive timebin metrics
    timebin_metrics_df = pd.concat([negative_timebin_metrics, pos_timebin_metrics])

    n_pos = timebin_metrics_df[timebin_metrics_df['label'] == 1].shape[0]
    n_neg = timebin_metrics_df[timebin_metrics_df['label'] == 0].shape[0]
    log_df = pd.DataFrame({'n_pos': [n_pos], 'n_neg': [n_neg]})

    if verbose:
        print(log_df)

    return timebin_metrics_df, log_df