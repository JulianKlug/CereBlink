import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Versions/4.1/Resources"
from pymer4.models import Lmer


def define_events_over_intensity_thresholds(df, intensity_threshold, parameter_name='systole', relative_time_column='relative_time'):
    """
    Define events by as all values above a given intensity threshold, for a given parameter
    Duration of events is recorded as the sum of duration of all consecutive measures
    Product is the sum of all products (intensity * duration) of all consecutive measures

    Arguments:
        - df: pandas Dataframe with all measures (expected columns of timing, duration and product)
        - intensity_threshold: threshold of insult intensity from which the events are defined
        - parameter_name: name of column with measured values in df

    Returns: events_df
    """


    # sum duration of subsequent rows above the intensity threshold
    df['exceeds_intensity_treshold'] = df[parameter_name] >= intensity_threshold
    
    # record all events of intensity > threshold and cumulative duration > threshold
    events_df = pd.DataFrame()
    # loop through all patients
    for pnr in tqdm(df['pNr'].unique()):
        pnr_df = df[df['pNr'] == pnr]
        # exclude values with negative delta_time (last time point of a patient)
        pnr_df = pnr_df[pnr_df['delta_time'] >= 0]

        pnr_df.sort_values(by=relative_time_column, inplace=True)
        
        # event starts where mask is True **and** the previous row was False (or N/A for first row)
        event_starts = pnr_df['exceeds_intensity_treshold'] & ~pnr_df['exceeds_intensity_treshold'].shift(fill_value=False)

        # Cumulative sum of event_starts gives a unique event number; set to 0 where mask is False
        pnr_df["event_id"] = event_starts.cumsum().where(pnr_df['exceeds_intensity_treshold'], 0).astype(int)
        pnr_events_df = pnr_df.groupby('event_id').agg({
                'pNr': 'first',
                relative_time_column: ['min', 'max'],
                'delta_time': 'sum',
                'product': 'sum',
                'DCI_YN_verified': 'first',
                'mrs_1y': 'first',
            }).reset_index()
        pnr_events_df.columns = ['event_id', 'pNr', 'event_first_measure_rel_time', 'event_last_measure_rel_time', 'event_duration', 'event_product',
                                    'DCI_YN_verified', 'mrs_1y']
        pnr_events_df['intensity_threshold'] = intensity_threshold
        pnr_events_df['parameter_name'] = parameter_name
        # drop the events with id 0
        pnr_events_df = pnr_events_df[pnr_events_df['event_id'] > 0]

        events_df = pd.concat([events_df, pnr_events_df], ignore_index=True)

    return events_df


def define_events_multiple_thresholds(df, intensity_thresholds, parameter_name='systole', relative_time_column='relative_time'):
    """
    Define events by as all values above a given intensity threshold, for a given parameter
    Duration of events is recorded as the sum of duration of all consecutive measures
    Product is the sum of all products (intensity * duration) of all consecutive measures

    Arguments:
        - df: pandas Dataframe with all measures (expected columns of timing, duration and product)
        - intensity_thresholds: list of thresholds of insult intensity from which the events are defined
        - parameter_name: name of column with measured values in df

    Returns: events_df
    """
    
    events_dfs = []
    for threshold in intensity_thresholds:
        events_df = define_events_over_intensity_thresholds(df, threshold, parameter_name, relative_time_column)
        events_dfs.append(events_df)

    return pd.concat(events_dfs, ignore_index=True)


def threshold_event_duration(events_df, duration_threshold):
    """
    Filter events based on a minimum duration threshold.
    
    Arguments:
        - events_df: DataFrame with events
        - duration_threshold: minimum duration of events to keep
    
    Returns: filtered_events_df
    """
    filtered_events_df = events_df[events_df['event_duration'] >= duration_threshold]
    filtered_events_df['duration_threshold'] = duration_threshold

    return filtered_events_df


def multiple_duration_thresholds(events_df, duration_thresholds):
    """
    Apply multiple duration thresholds to filter events.
    
    Arguments:
        - events_df: DataFrame with events
        - duration_thresholds: list of duration thresholds
    
    Returns: filtered_events_df
    """
    filtered_events_dfs = []
    for threshold in duration_thresholds:
        filtered_events_df = threshold_event_duration(events_df, threshold)
        filtered_events_dfs.append(filtered_events_df)
    
    return pd.concat(filtered_events_dfs, ignore_index=True)


def count_events(events_df):
    """
    Count events for each pNr, intensity threshold, and duration threshold.
        - construct a DataFrame with a count of events for each pnr and intensity threshold and duration threshold
        - keep the columns: pNr, intensity_threshold, duration_threshold, event_count, DCI_YN_verified, mrs_1y
    
    Arguments:
        - events_df: DataFrame with events
    
    Returns: counts_df
    """
    counts_df = events_df.groupby(['pNr', 'intensity_threshold', 'duration_threshold']).agg(
                                    event_count=('event_id', 'nunique'),
                                    DCI_YN_verified=('DCI_YN_verified', 'first'),
                                    mrs_1y=('mrs_1y', 'first')).reset_index()
    return counts_df


# for every intensity threshold and duration trheshold combination, compute the pearson correlation coefficient between number of events and mRS_1y
def event_count_to_mrs_correlation(events_df):
    """
    Compute the Pearson correlation coefficient between event counts and mRS_1y for each intensity and duration threshold.
    
    Arguments:
        - events_df: DataFrame with event counts
    
    Returns: correlation_df
    """
    correlation_results = []
    
    for (intensity_threshold, duration_threshold), group in events_df.groupby(['intensity_threshold', 'duration_threshold']):
        temp_df = group.copy()
        temp_df.dropna(subset=['event_count', 'mrs_1y'], inplace=True)
        if temp_df.empty:
            continue
        if len(temp_df) > 1:  # Ensure there are enough data points to compute correlation
            corr, p_value = stats.pearsonr(temp_df['event_count'], temp_df['mrs_1y'])
            correlation_results.append({
                'intensity_threshold': intensity_threshold,
                'duration_threshold': duration_threshold,
                'correlation_coefficient': corr,
                'p_value': p_value
            })
    
    return pd.DataFrame(correlation_results)


# for every intensity threshold and duration trheshold combination, compute the pearson correlation coefficient between event_product and mRS_1y
def event_product_to_mrs_correlation(events_df):
    """
    Compute the Pearson correlation coefficient between event product and mRS_1y for each intensity and duration threshold.
    
    Arguments:
        - events_df: DataFrame with event products
    
    Returns: correlation_df
    """
    correlation_results = []
    
    for (intensity_threshold, duration_threshold), group in events_df.groupby(['intensity_threshold', 'duration_threshold']):
        temp_df = group.copy()
        temp_df.dropna(subset=['event_product', 'mrs_1y'], inplace=True)
        if temp_df.empty:
            continue
        if len(temp_df) > 1:  # Ensure there are enough data points to compute correlation
            corr, p_value = stats.pearsonr(temp_df['event_product'], temp_df['mrs_1y'])
            correlation_results.append({
                'intensity_threshold': intensity_threshold,
                'duration_threshold': duration_threshold,
                'correlation_coefficient': corr,
                'p_value': p_value
            })
    
    return pd.DataFrame(correlation_results)


def event_count_to_DCI_coefficient(events_df, DCI_column='DCI_YN_verified'):
    """
    Compute the logistic regression coefficient between event counts and DCI_YN_verified for each intensity and duration threshold.
    Arguments:
        - events_df: DataFrame with event counts
        - DCI_column: column name for DCI_YN_verified (default is 'DCI_YN_verified')
    Returns: coefficient_df
    """
    coefficient_results = []
    
    for (intensity_threshold, duration_threshold), group in events_df.groupby(['intensity_threshold', 'duration_threshold']):
        temp_df = group.copy()
        temp_df.dropna(subset=['event_count', DCI_column], inplace=True)
        
        # Skip if too few data points
        if len(temp_df) <= 1:
            continue
            
        # Check for zero variance
        if temp_df['event_count'].std() == 0:
            continue
            
        try:
            log_model = sm.Logit(temp_df[DCI_column],
                                 sm.add_constant(temp_df['event_count']))
            log_model_result = log_model.fit(disp=0, method='bfgs')  # Try different solver
            coefficient_results.append({
                'intensity_threshold': intensity_threshold,
                'duration_threshold': duration_threshold,
                'coefficient': log_model_result.params['event_count'],
                'p_value': log_model_result.pvalues['event_count'],
                'n_samples': len(temp_df)
            })
        except Exception as e:
            print(f"Error at threshold {intensity_threshold}/{duration_threshold}: {e}")

    
    return pd.DataFrame(coefficient_results)


def event_product_to_DCI_coefficient(events_df, DCI_column='DCI_YN_verified', use_mixed_effects=False):
    """
    Compute the logistic regression coefficient between event product and DCI_YN_verified for each intensity and duration threshold.
    
    Arguments:
        - events_df: DataFrame with event products
        - DCI_column: column name for DCI_YN_verified (default is 'DCI_YN_verified')
    
    Returns: coefficient_df
    """
    coefficient_results = []
    
    for (intensity_threshold, duration_threshold), group in events_df.groupby(['intensity_threshold', 'duration_threshold']):
        temp_df = group.copy()
        temp_df.dropna(subset=['event_product', DCI_column], inplace=True)
        
        # Skip if too few data points
        if len(temp_df) <= 1:
            continue
            
        # Check for zero variance
        if temp_df['event_product'].std() == 0:
            continue

        # rescale event_product by mean and standard deviation
        temp_df['event_product'] = (temp_df['event_product'] - temp_df['event_product'].mean()) / temp_df['event_product'].std()

            
        try:
            if not use_mixed_effects:
                log_model = sm.Logit(temp_df[DCI_column],
                                    sm.add_constant(temp_df['event_product']))
                log_model_result = log_model.fit(disp=0, method='bfgs')  # Try different solver
                coefficient = log_model_result.params['event_product']
                p_value = log_model_result.pvalues['event_product']
            else:
                from rpy2.robjects.packages import importr
                from rpy2.robjects import pandas2ri
                import rpy2.robjects as ro

                stats = importr('stats')
                lme4 = importr('lme4')
                base = importr('base')
                lmerT = importr('lmerTest')


                with (ro.default_converter + pandas2ri.converter).context():
                    metric_r_df = ro.conversion.get_conversion().py2rpy(temp_df)
                metric = 'event_product'

                model = lmerT.lmer(f"{DCI_column}  ~ {metric}  + (1|pNr)",
                                data=metric_r_df)
                coeffs = base.summary(model).rx2('coefficients')
                indices = np.asarray(list(coeffs.names)[0])
                column_names = np.asarray(list(coeffs.names)[1])

                with (ro.default_converter + pandas2ri.converter).context():
                    coeffs_df = pd.DataFrame(ro.conversion.get_conversion().rpy2py(coeffs),
                                                index=indices, columns=column_names)

                warnings = base.summary(model).rx2('warnings')
                # check if warnings is null
                if warnings != ro.rinterface.NULL:
                    # r_model_warnings_df = pd.concat(
                    #     [r_model_warnings_df, pd.DataFrame({'metric': [metric], 'warning': [warnings]})])
                    print(f"Warning for treshold {intensity_threshold}/{duration_threshold}: {warnings}")

                coefficient = coeffs_df.loc[metric, 'Estimate']
                p_value = coeffs_df.loc[metric, 'Pr(>|t|)']


            coefficient_results.append({
                'intensity_threshold': intensity_threshold,
                'duration_threshold': duration_threshold,
                'coefficient': coefficient,
                'p_value': p_value,
                'n_samples': len(temp_df)
            })
        except Exception as e:
            print(f"Error at threshold {intensity_threshold}/{duration_threshold}: {e}")

    
    return pd.DataFrame(coefficient_results)