import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
#os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Versions/4.1/Resources"
#from pymer4.models import Lmer


def define_events_over_intensity_thresholds(df, intensity_threshold, parameter_name='systole', relative_time_column='relative_time', bp_focus='hyptertension'):
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
    if bp_focus == 'hypertension':
        df['exceeds_intensity_treshold'] = df[parameter_name] >= intensity_threshold
    elif bp_focus == 'hypotension':
        df['exceeds_intensity_treshold'] = df[parameter_name] <= intensity_threshold
    
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


def define_events_multiple_thresholds(df, intensity_thresholds, parameter_name='systole', relative_time_column='relative_time',bp_focus='hyptertension'):
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
        events_df = define_events_over_intensity_thresholds(df, threshold, parameter_name, relative_time_column, bp_focus)
        events_dfs.append(events_df)

    return pd.concat(events_dfs, ignore_index=True)


def threshold_event_duration(events_df, duration_threshold, parameter_name):
    """
    Filter events based on a minimum duration threshold.
    
    Arguments:
        - events_df: DataFrame with events
        - duration_threshold: minimum duration of events to keep
    
    Returns: filtered_events_df
    """
    filtered_events_df = events_df[events_df['event_duration'] >= duration_threshold]
    filtered_events_df['duration_threshold'] = duration_threshold


    #exclude bloodpressure values which appear in less than 10 person
   
    if filtered_events_df.pNr.nunique()<10:
       filtered_events_df=filtered_events_df[0:0]

    return filtered_events_df


def multiple_duration_thresholds(events_df, bp_parameter, duration_thresholds):
    """
    Apply multiple duration thresholds to filter events.
    
    Arguments:
        - events_df: DataFrame with events
        - duration_thresholds: list of duration thresholds
    
    Returns: filtered_events_df
    """
    filtered_events_dfs = []
    for threshold in duration_thresholds:
        filtered_events_df = threshold_event_duration(events_df, threshold, parameter_name='systole')
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


def total_event_duration(events_df):
    """
    Compute the total event duration for each pNr, intensity threshold, and duration threshold.
        - construct a DataFrame with total event duration for each pnr and intensity threshold and duration threshold
        - keep the columns: pNr, intensity_threshold, duration_threshold, total_event_duration, DCI_YN_verified, mrs_1y

    Arguments:
        - events_df: DataFrame with events

    Returns: duration_df
    """
    duration_df = events_df.groupby(['pNr', 'intensity_threshold', 'duration_threshold']).agg(
                                    total_event_duration=('event_duration', 'sum'),
                                    DCI_YN_verified=('DCI_YN_verified', 'first'),
                                    mrs_1y=('mrs_1y', 'first')).reset_index()
    return duration_df


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
        if len(temp_df) > 2:  # Ensure there are enough data points to compute correlation
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
        if len(temp_df) > 2:  # Ensure there are enough data points to compute correlation
            corr, p_value = stats.pearsonr(temp_df['event_product'], temp_df['mrs_1y'])
            correlation_results.append({
                'intensity_threshold': intensity_threshold,
                'duration_threshold': duration_threshold,
                'correlation_coefficient': corr,
                'p_value': p_value
            })
    
    return pd.DataFrame(correlation_results)


def event_relative_duration_to_mrs_correlation(events_df, relative_duration_column='total_event_duration_proportion'):
    """
    Compute the Pearson correlation coefficient between event relative duration and mRS_1y for each intensity and duration threshold.

    Arguments:
        - events_df: DataFrame with event relative durations

    Returns: correlation_df
    """
    correlation_results = []
    for (intensity_threshold, duration_threshold), group in events_df.groupby(['intensity_threshold', 'duration_threshold']):
        temp_df = group.copy()
        temp_df.dropna(subset=[relative_duration_column, 'mrs_1y'], inplace=True)
        if temp_df.empty:
            continue
        if len(temp_df) > 2:
            corr, p_value = stats.pearsonr(temp_df[relative_duration_column], temp_df['mrs_1y'])
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


def total_correlated_event_counts(event_counts_df, correlation_df):
    """
    Calculate the total number of positively and negatively correlated events.
    """
    

    event_counts_df['positively_correlated'] = 0
    event_counts_df['negatively_correlated'] = 0
    for idx, row in correlation_df.iterrows():
        if row['correlation_coefficient'] > 0:
            event_counts_df.loc[
                (event_counts_df['intensity_threshold'] == row['intensity_threshold']) &
                (event_counts_df['duration_threshold'] == row['duration_threshold']),
                'positively_correlated'] = 1
        elif row['correlation_coefficient'] < 0:
            event_counts_df.loc[
                (event_counts_df['intensity_threshold'] == row['intensity_threshold']) &
                (event_counts_df['duration_threshold'] == row['duration_threshold']),
                'negatively_correlated'] = 1
            
    event_counts_df['positively_correlated_event_count'] = event_counts_df['positively_correlated'] * event_counts_df['event_count']
    event_counts_df['negatively_correlated_event_count'] = event_counts_df['negatively_correlated'] * event_counts_df['event_count']
    
    return event_counts_df.groupby('pNr').agg({
            'positively_correlated_event_count': 'sum',
            'negatively_correlated_event_count': 'sum',
            'DCI_YN_verified': 'first',
            'mrs_1y': 'first'
        }).reset_index()

def total_correlated_event_duration(event_df, correlation_df, threshold=0):
    """
    Calculate the total number of positively and negatively correlated events.
    """
    if 'correlation_coefficient' in correlation_df.columns:
        coefficient_name = 'correlation_coefficient'
    elif 'coefficient' in correlation_df.columns:
        coefficient_name = 'coefficient'
    else:
        raise ValueError("correlation_df must contain either 'correlation_coefficient' or 'coefficient' column.")
    
    event_df['positively_correlated'] = 0
    event_df['negatively_correlated'] = 0
    for idx, row in correlation_df.iterrows():
        if row[coefficient_name] > threshold:
            event_df.loc[
                (event_df['intensity_threshold'] == row['intensity_threshold']) &
                (event_df['duration_threshold'] == row['duration_threshold']),
                'positively_correlated'] = 1
        elif row[coefficient_name] < -1 * threshold:
            event_df.loc[
                (event_df['intensity_threshold'] == row['intensity_threshold']) &
                (event_df['duration_threshold'] == row['duration_threshold']),
                'negatively_correlated'] = 1
            
    event_df['positively_correlated_event_duration'] = event_df['positively_correlated'] * event_df['event_duration']
    event_df['negatively_correlated_event_duration'] = event_df['negatively_correlated'] * event_df['event_duration']

    return event_df.groupby('pNr').agg({
            'positively_correlated_event_duration': 'sum',
            'negatively_correlated_event_duration': 'sum',
            'DCI_YN_verified': 'first',
            'mrs_1y': 'first'
        }).reset_index()



def relative_duration_in_correlated_events(duration_thresholded_events_df, event_count_correlation_df, monitoring_duration_df, threshold=0):
    no_duplicates_duration_thresholded_events_df = duration_thresholded_events_df.groupby(['pNr', 'event_id']).agg({
    'parameter_name': 'first',
    'event_first_measure_rel_time': 'first',
    'event_last_measure_rel_time': 'first',
    'event_duration': 'max',
    'event_product': 'max',
    'intensity_threshold': 'max',
    'duration_threshold': 'max',
    'DCI_YN_verified': 'first',
    'mrs_1y': 'first'
    }).reset_index()

    event_duration_with_correlation_df = total_correlated_event_duration(no_duplicates_duration_thresholded_events_df, event_count_correlation_df,
                                                                                        threshold=threshold)

    event_duration_with_correlation_df = event_duration_with_correlation_df.merge(
        monitoring_duration_df[['pNr', 'monitoring_duration']],
        on='pNr',
        how='left'
    )

    event_duration_with_correlation_df['positively_correlated_event_proportion_of_monitoring_duration'] = event_duration_with_correlation_df['positively_correlated_event_duration'] / event_duration_with_correlation_df['monitoring_duration']
    event_duration_with_correlation_df['negatively_correlated_event_proportion_of_monitoring_duration'] = event_duration_with_correlation_df['negatively_correlated_event_duration'] / event_duration_with_correlation_df['monitoring_duration']

    return event_duration_with_correlation_df


def decision_boundary_analysis(duration_thresholded_events_df, event_count_correlation_df, monitoring_duration_df, covariate_df, bp_focus='hypertension', 
                               correlation_threshold=0,
                                       outcome='mrs_1y', reg_type='ordinal',
                                       multivariable=True, verbose=False):
    """
    Uni and multivariable analysis of decision boundary (proportion of duration in positively or negatively correlated zone)

    Arguments:
    duration_thresholded_events_df : DataFrame
        DataFrame containing duration thresholded events.
    event_count_correlation_df : DataFrame
        DataFrame containing event count correlations.
    monitoring_duration_df : DataFrame
        DataFrame containing monitoring durations.
    covariate_df : DataFrame
        DataFrame containing covariates.
    correlation_threshold : float
        The correlation threshold to use.
    outcome : str
        The outcome variable to analyze.
    reg_type : str
        The type of regression to use ('ordinal' or 'log').
    multivariable : bool
        Whether to include multivariable analysis.
    verbose : bool
        Whether to print detailed output.
    Returns:
    pos_event_duration_result : statsmodels result object
    neg_event_duration_result : statsmodels result object
    pos_event_duration_result_multivariable : statsmodels result object or None
    neg_event_duration_result_multivariable : statsmodels result object or None
    """
    relative_duration_in_correlated_events_df = relative_duration_in_correlated_events(
        duration_thresholded_events_df,
        event_count_correlation_df,
        monitoring_duration_df,
        threshold=correlation_threshold
    )

    # Univariate association of duration with mRS_1y (ordinal regression)
    temp_df = relative_duration_in_correlated_events_df[['positively_correlated_event_proportion_of_monitoring_duration', 'negatively_correlated_event_proportion_of_monitoring_duration', outcome]].dropna()

    if reg_type == 'ordinal': 
        pos_event_duration_model = OrderedModel(
            temp_df[outcome],
            temp_df[['positively_correlated_event_proportion_of_monitoring_duration']],
            distr='logit'
        )
        neg_event_duration_model = OrderedModel(
            temp_df[outcome],
            temp_df[['negatively_correlated_event_proportion_of_monitoring_duration']],
            distr='logit'
        )
        pos_event_duration_result = pos_event_duration_model.fit(method='bfgs')
        neg_event_duration_result = neg_event_duration_model.fit(method='bfgs')


    elif reg_type == 'log':
        pos_event_duration_model = sm.Logit(temp_df[outcome],
                                    sm.add_constant(temp_df[['positively_correlated_event_proportion_of_monitoring_duration']]))
        pos_event_duration_result = pos_event_duration_model.fit(disp=0, method='bfgs')  

        neg_event_duration_model = sm.Logit(temp_df[outcome],
                                    sm.add_constant(temp_df[['negatively_correlated_event_proportion_of_monitoring_duration']]))
        neg_event_duration_result = neg_event_duration_model.fit(disp=0, method='bfgs')  

    else:
        raise ValueError(f'Unsupported regression type: {reg_type}. Supported types are "ordinal" and "log".')

    if verbose:
        print("Positive Event Duration Model Summary:")
        print(pos_event_duration_result.summary())
        print("Negative Event Duration Model Summary:")
        print(neg_event_duration_result.summary())

    # Mulivariable
    if multivariable:
        # mutlivariable model with Age, WFNS, Fisher_Score, Coiling, Clipping
        temp_df = relative_duration_in_correlated_events_df.merge(
            covariate_df[['pNr', 'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']],
            on='pNr',
            how='left'
        )
        temp_df = temp_df[['positively_correlated_event_proportion_of_monitoring_duration',
                        'negatively_correlated_event_proportion_of_monitoring_duration',
                        outcome, 'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']].dropna()

        temp_df['Age'] = pd.to_numeric(temp_df['Age'], errors='coerce')
        temp_df['WFNS'] = pd.to_numeric(temp_df['WFNS'], errors='coerce')
        temp_df['Fisher_Score'] = pd.to_numeric(temp_df['Fisher_Score'], errors='coerce')
        temp_df['Coiling'] = pd.to_numeric(temp_df['Coiling'], errors='coerce')
        temp_df['Clipping'] = pd.to_numeric(temp_df['Clipping'], errors='coerce')

        if reg_type == 'ordinal':
            pos_event_duration_model_multivariable = OrderedModel(
                temp_df[outcome],
                temp_df[['positively_correlated_event_proportion_of_monitoring_duration',
                        'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']],
                distr='logit'
            )
            neg_event_duration_model_multivariable = OrderedModel(
                temp_df[outcome],
                temp_df[['negatively_correlated_event_proportion_of_monitoring_duration',
                        'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']],
                distr='logit'
            )
            pos_event_duration_result_multivariable = pos_event_duration_model_multivariable.fit(method='bfgs')
            neg_event_duration_result_multivariable = neg_event_duration_model_multivariable.fit(method='bfgs')
        elif reg_type == 'log':
            pos_event_duration_model_multivariable = sm.Logit(temp_df[outcome],
                                    sm.add_constant(temp_df[['positively_correlated_event_proportion_of_monitoring_duration',
                                                            'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']]))
            pos_event_duration_result_multivariable = pos_event_duration_model_multivariable.fit(disp=0, method='bfgs')  

            neg_event_duration_model_multivariable = sm.Logit(temp_df[outcome],
                                    sm.add_constant(temp_df[['negatively_correlated_event_proportion_of_monitoring_duration',
                                                            'Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']]))
            neg_event_duration_result_multivariable = neg_event_duration_model_multivariable.fit(disp=0, method='bfgs')
        else:
            raise ValueError(f'Unsupported regression type: {reg_type}. Supported types are "ordinal" and "log".')
        
        if verbose:
            print("Positive Event Duration Multivariable Model Summary:")
            print(pos_event_duration_result_multivariable.summary())
            print("Negative Event Duration Multivariable Model Summary:")
            print(neg_event_duration_result_multivariable.summary())

        return pos_event_duration_result, neg_event_duration_result, pos_event_duration_result_multivariable, neg_event_duration_result_multivariable
    else: 
        return pos_event_duration_result, neg_event_duration_result, None, None
    

def save_regression_analysis_results_to_csv(regression_result, output_dir, filename_root):
        fit_as_html = regression_result.summary().tables[0].as_html()
        results_as_html = regression_result.summary().tables[1].as_html()
        pd.read_html(fit_as_html, header=0, index_col=0)[0].to_csv(os.path.join(output_dir, f'{filename_root}_regression_fit.csv'))
        pd.read_html(results_as_html, header=0, index_col=0)[0].to_csv(os.path.join(output_dir, f'{filename_root}_regression_results.csv')) 

def save_decision_boundary_analysis_results_to_csv(decision_boundary_results, output_dir, filename_root):
    pos_event_duration_result, neg_event_duration_result, pos_event_duration_result_multivariable, neg_event_duration_result_multivariable = decision_boundary_results
    save_regression_analysis_results_to_csv(pos_event_duration_result, output_dir, f'{filename_root}_pos_event_duration')
    save_regression_analysis_results_to_csv(neg_event_duration_result, output_dir, f'{filename_root}_neg_event_duration')
    save_regression_analysis_results_to_csv(pos_event_duration_result_multivariable, output_dir, f'{filename_root}_pos_event_duration_multivariable')
    save_regression_analysis_results_to_csv(neg_event_duration_result_multivariable, output_dir, f'{filename_root}_neg_event_duration_multivariable')
       

def plot_event_correlation_heatmap(correlation_df: pd.DataFrame, coefficient_name: str = 'correlation_coefficient', step_size:int = 10):
    fig, ax = plt.subplots(figsize=(12, 8))

    annotations = True if step_size >= 10 else False

    ax = sns.heatmap(correlation_df.pivot_table(
        index='duration_threshold',
        columns='intensity_threshold',
        values=coefficient_name
    ).reindex(index=sorted(correlation_df['duration_threshold'].unique(), reverse=True)),
        annot=annotations, cmap='seismic', center=0, ax=ax)
    
    return fig