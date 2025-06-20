import getpass
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from utils.utils import load_encrypted_xlsx, ensure_dir
from bp.bp_burden.analysis_utils import count_events, define_events_multiple_thresholds, event_count_to_mrs_correlation, multiple_duration_thresholds, decision_boundary_analysis, plot_event_correlation_heatmap, save_decision_boundary_analysis_results_to_csv, event_count_to_DCI_coefficient


def event_burden_analysis(working_df, intensity_threshold_range, intensity_threshold_step, duration_range, duration_step, 
                          bp_parameter, outcome, period_name,
                          output_dir, monitoring_duration_df, main_df, 
                          verbose=False, correlation_threshold=0):
    if outcome == 'mrs_1y':
        reg_type = 'ordinal'
        coefficient_name = 'correlation_coefficient'
    elif outcome == 'DCI_YN_verified':
        reg_type = 'log'
        coefficient_name = 'coefficient'
    
    events_df = define_events_multiple_thresholds(working_df,
                                            intensity_thresholds=range(intensity_threshold_range[0], intensity_threshold_range[1] + 1, intensity_threshold_step),
                                            parameter_name=bp_parameter)

    duration_thresholded_events_df = multiple_duration_thresholds(events_df,
                                                                    duration_thresholds=range(duration_range[0], duration_range[1] + 1, duration_step))

    event_counts_df = count_events(duration_thresholded_events_df)

    if outcome == 'mrs_1y':
        association_df = event_count_to_mrs_correlation(event_counts_df)
    elif outcome == 'DCI_YN_verified':
        association_df = event_count_to_DCI_coefficient(event_counts_df)

    # Create figure
    fig = plot_event_correlation_heatmap(association_df, coefficient_name=coefficient_name, step_size=min(intensity_threshold_step, duration_step))
    fig.suptitle(f'{bp_parameter} event counts vs {outcome} in {period_name}')
    fig.savefig(os.path.join(output_dir, f'{bp_parameter}_event_counts_vs_{outcome}_in_{period_name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)

    decision_boundary_analysis_results = decision_boundary_analysis(duration_thresholded_events_df, association_df, monitoring_duration_df, main_df, verbose=verbose,
                                                                        correlation_threshold=correlation_threshold,
                                                                        outcome=outcome, reg_type=reg_type)
    save_decision_boundary_analysis_results_to_csv(decision_boundary_analysis_results, output_dir, f'{bp_parameter}_{outcome}_t{correlation_threshold}_{period_name}_decision_boundary_analysis')
    
    return decision_boundary_analysis_results


def bp_events_analysis_pipeline(
        registry_data_path: str,
        nor_annotated_bp_data_path: str,
        correspondance_data_path: str,
        outcome_data_path: str,
        output_dir: str,
        filter_noradrenaline: bool = False,
        bp_parameter: str = 'systole',
        outcome: str = 'mrs_1y',
        intensity_threshold_range: tuple = (140, 220),
        intensity_threshold_step: int = 10,
        duration_range: tuple = (0, 180),
        duration_step: int = 10,
        correlation_threshold: float = 0,
        registry_password: str = None,
        outcome_password: str = None,
        verbose: bool = False):
    
    # check outcome is in ['mrs_1y', 'DCI_YN_verified']
    if outcome not in ['mrs_1y', 'DCI_YN_verified']:
        raise ValueError(f'Invalid outcome: {outcome}. Must be one of ["mrs_1y", "DCI_YN_verified"].')
    # bp_parameter must be in ['systole', 'diastole', 'mitteldruck']
    if bp_parameter not in ['systole', 'diastole', 'mitteldruck']:
        raise ValueError(f'Invalid bp_parameter: {bp_parameter}. Must be one of ["systole", "diastole", "mitteldruck"].')
    
    ensure_dir(output_dir)

    registry_df = load_encrypted_xlsx(registry_data_path, password=registry_password)
    outcome_df = load_encrypted_xlsx(outcome_data_path, password=outcome_password)
    nor_annotated_bp_df = pd.read_csv(nor_annotated_bp_data_path)

    if filter_noradrenaline:
        bp_df = nor_annotated_bp_df[nor_annotated_bp_df['noradrenaline_concomitant'] == 0]
    else:
        bp_df = nor_annotated_bp_df

    registry_pdms_correspondance_df = pd.read_csv(correspondance_data_path)

    # drop duplicates 
    bp_df = bp_df.drop_duplicates(subset=['pNr', 
                                        'systole',
                                        'diastole',
                                        'mitteldruck',
                                        'timeBd'])

    registry_df.drop_duplicates(inplace=True)
    registry_df.dropna(subset=['SOS-CENTER-YEAR-NO.', 'Name', 'Date_birth'], inplace=True)

    bp_df=bp_df.merge(registry_pdms_correspondance_df, how='left', on='pNr')

    bp_df['Date_birth']=pd.to_datetime(bp_df['Date_birth'], format='%d.%m.%Y')
    outcome_df['Date_birth']=pd.to_datetime(outcome_df['Date_birth'])

    outcome_df["mRS_FU_1y"]=pd.to_numeric(outcome_df['mRS_FU_1y'], errors='coerce')

    for pnr in tqdm(bp_df["pNr"].unique()):
        sos_center_nr = bp_df[bp_df["pNr"] == pnr]["SOS-CENTER-YEAR-NO."].values[0]
        name = bp_df[bp_df["pNr"] == pnr]["JoinedName"].values[0]
        date_birth = bp_df[bp_df["pNr"] == pnr]["Date_birth"].values[0]
        mrs_values = outcome_df[(outcome_df["SOS-CENTER-YEAR-NO."] == sos_center_nr) &
                            (outcome_df["Name"] == name) &
                            (outcome_df["Date_birth"] == date_birth)]["mRS_FU_1y"]
        if len(mrs_values) == 0:
            mrs = np.nan
        else:
            mrs = mrs_values.values[0]

        bp_df.loc[bp_df["pNr"] == pnr, "mrs_1y"] = mrs

    # for each pNr in bp_df, get durtation of monitoring by last_measure - first_measure
    bp_df['timeBd']=pd.to_datetime(bp_df['timeBd'], format='%Y-%m-%d %H:%M:%S.%f')
    monitoring_duration_df = bp_df.groupby('pNr')['timeBd'].agg(['min', 'max']).reset_index()
    monitoring_duration_df['monitoring_duration'] = (monitoring_duration_df['max'] - monitoring_duration_df['min']).dt.total_seconds() / 60  # convert to minutes

    main_df = bp_df.merge(registry_df, 
                        left_on=['SOS-CENTER-YEAR-NO.', 'JoinedName', 'Date_birth'], 
                        right_on=['SOS-CENTER-YEAR-NO.', 'Name', 'Date_birth'], 
                        how='left')
    
    # compute timings
    main_df['Date_DCI_ischemia_first_image'] = pd.to_datetime(main_df['Date_DCI_ischemia_first_image'], errors='coerce', format='%Y-%m-%d')
    main_df['Time_DCI_ischemia_first_image'] = pd.to_datetime(main_df['Time_DCI_ischemia_first_image'], errors='coerce', format='%H:%M:%S')

    main_df['Date_DCI_infarct_first_image'] = pd.to_datetime(main_df['Date_DCI_infarct_first_image'], errors='coerce', format='%Y-%m-%d')
    main_df['Date_DCI_infarct_first_image'] = pd. to_datetime(main_df['Date_DCI_infarct_first_image'],errors='coerce', format='%H:%M:%S')

    main_df['timestamp_ischemia'] = pd.to_datetime(
        main_df['Date_DCI_ischemia_first_image'].astype(str) + ' ' + main_df['Time_DCI_ischemia_first_image'].astype(str),
        errors='coerce'
    )

    main_df['timestamp_infarction'] =  pd.to_datetime(
        main_df['Date_DCI_infarct_first_image'].astype(str) + ' ' + main_df['Time_DCI_infarct_first_image'].astype(str),
        errors='coerce'
    )

    main_df['timeBd']=pd.to_datetime(main_df['timeBd'], format='%Y-%m-%d %H:%M:%S.%f')

    main_df['timeBd'] = main_df['timeBd'].dt.tz_localize(None)
    main_df['timestamp_ischemia'] = main_df['timestamp_ischemia'].dt.tz_localize(None)

    main_df['time_difference_ischemia']=main_df['timestamp_ischemia'] - main_df['timeBd']
    main_df['time_difference_ischemia']=main_df['time_difference_ischemia'].dt.total_seconds() / 60

    main_df['time_difference_infarction']=main_df['timestamp_infarction']-main_df['timeBd']
    main_df['time_difference_infarction']=main_df['time_difference_infarction'].dt.total_seconds() / 60

    main_df = main_df.sort_values(by=['pNr', 'timeBd'], ascending=True)

    main_df['T0'] = main_df.groupby('pNr')['timeBd'].transform('min')
    main_df['relative_time'] = main_df['timeBd'] - main_df['T0']
    main_df['relative_time'] = main_df['relative_time'].dt.total_seconds() / 60
    main_df['relative_time'] = pd.to_numeric(main_df['relative_time'], errors='coerce')

    main_df['first_Th_relative_date'] = (pd.to_datetime(main_df['Date_First_Th']) - main_df['T0']).dt.total_seconds() / 60

    # compute pressure time product
    main_df['delta_time'] = main_df['timeBd'].shift(-1) - main_df['timeBd']
    main_df['delta_time'] = main_df['delta_time'].dt.total_seconds() / 60
    main_df['product'] = main_df['delta_time'] * main_df[bp_parameter]

    working_df = main_df[['relative_time', 'pNr', 'delta_time', 'systole', 'product', 'DCI_YN_verified', 'mrs_1y', 'first_Th_relative_date']]


    # Analysis
    # First 24h
    # analysis for first 24 hours of monitoring
    working_df_in_first_24h_monitoring = working_df[working_df['relative_time'] <= 24 * 60]  # 24 hours in minutes
    _ = event_burden_analysis(working_df_in_first_24h_monitoring, intensity_threshold_range, intensity_threshold_step, duration_range, duration_step,
                                    bp_parameter, outcome, 'first_24h',
                                    output_dir, monitoring_duration_df, main_df,
                                    verbose=verbose, correlation_threshold=correlation_threshold)

    # analysis for 24h-to-end of monitoring
    working_df_after_24h_monitoring = working_df[working_df['relative_time'] > 24 * 60]  # 24 hours in minutes
    _ = event_burden_analysis(working_df_after_24h_monitoring, intensity_threshold_range, intensity_threshold_step, duration_range, duration_step,
                                    bp_parameter, outcome, 'after_24h',
                                    output_dir, monitoring_duration_df, main_df,
                                    verbose=verbose, correlation_threshold=correlation_threshold)

    # before aneurysm treatment
    working_df_before_aneurym_secured = working_df[working_df['relative_time'] < (working_df['first_Th_relative_date'] + 24 * 60)]  # 24 hours in minutes
    _ = event_burden_analysis(working_df_before_aneurym_secured, intensity_threshold_range, intensity_threshold_step, duration_range, duration_step,
                                    bp_parameter, outcome, 'before_aneurysm_secured',
                                    output_dir, monitoring_duration_df, main_df,
                                    verbose=verbose, correlation_threshold=correlation_threshold)
    

    # after aneurysm treatment
    working_df_after_aneurym_secured = working_df[working_df['relative_time'] >= (working_df['first_Th_relative_date'] + 24 * 60)]  # 24 hours in minutes
    _ = event_burden_analysis(working_df_after_aneurym_secured, intensity_threshold_range, intensity_threshold_step, duration_range, duration_step,
                                    bp_parameter, outcome, 'after_aneurysm_secured',
                                    output_dir, monitoring_duration_df, main_df,
                                    verbose=verbose, correlation_threshold=correlation_threshold)
    

def all_outcomes_bp_events_analysis_pipeline(
        registry_data_path: str,
        nor_annotated_bp_data_path: str,
        correspondance_data_path: str,
        outcome_data_path: str,
        output_dir: str,
        filter_noradrenaline: bool = False,
        bp_parameter: str = 'systole',
        outcomes:  list = ['mrs_1y', 'DCI_YN_verified'],
        intensity_threshold_range: tuple = (140, 220),
        intensity_threshold_step: int = 10,
        duration_range: tuple = (0, 180),
        duration_step: int = 10,
        correlation_threshold: float = 0,
        verbose: bool = False):

    registry_password = getpass.getpass("Enter password for registry data: ")
    outcome_password = getpass.getpass("Enter password for outcome data: ")

    for outcome in outcomes:
        if verbose:
            print(f'Running analysis for outcome: {outcome}')

        outcome_dir =  os.path.join(output_dir, outcome)
        ensure_dir(outcome_dir)

        bp_events_analysis_pipeline(
            registry_data_path=registry_data_path,
            nor_annotated_bp_data_path=nor_annotated_bp_data_path,
            correspondance_data_path=correspondance_data_path,
            outcome_data_path=outcome_data_path,
            output_dir=outcome_dir,
            filter_noradrenaline=filter_noradrenaline,
            bp_parameter=bp_parameter,
            outcome=outcome,
            intensity_threshold_range=intensity_threshold_range,
            intensity_threshold_step=intensity_threshold_step,
            duration_range=duration_range,
            duration_step=duration_step,
            correlation_threshold=correlation_threshold,
            registry_password=registry_password,
            outcome_password=outcome_password,
            verbose=verbose
        )


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='Run BP events analysis pipeline.')
    parser.add_argument('-r', '--registry_data_path', type=str, required=True,
                        help='Path to the registry data file (encrypted Excel).')
    parser.add_argument('-b', '--nor_annotated_bp_data_path', type=str, required=True,
                        help='Path to the NOR annotated BP data file (CSV).')
    parser.add_argument('-c', '--correspondance_data_path', type=str, required=True,
                        help='Path to the registry-PDMS correspondance data file (CSV).')
    parser.add_argument('-o', '--outcome_data_path', type=str, required=True,
                        help='Path to the outcome data file (encrypted Excel).')
    parser.add_argument('-d', '--output_dir', type=str, required=True,
                        help='Directory to save the output results.')
    parser.add_argument('--filter_noradrenaline', action='store_true',
                        help='Whether to filter out records with noradrenaline concomitant use.')
    parser.add_argument('--bp_parameter', type=str, default='systole',
                        choices=['systole', 'diastole', 'mitteldruck'],
                        help='Blood pressure parameter to analyze.')
    parser.add_argument('--outcomes', type=str, nargs='+', default=['mrs_1y', 'DCI_YN_verified'],
                        help='List of outcomes to analyze. Default is ["mrs_1y", "DCI_YN_verified"].')
    parser.add_argument('--intensity_threshold_range', type=int, nargs=2, default=(140, 220),
                        help='Range of intensity thresholds for event definition (min, max). Default is (140, 220).')
    parser.add_argument('--intensity_threshold_step', type=int, default=10,
                        help='Step size for intensity thresholds. Default is 10.')
    parser.add_argument('--duration_range', type=int, nargs=2, default=(0, 180),
                        help='Range of duration thresholds for event definition (min, max). Default is (0, 180).')
    parser.add_argument('--duration_step', type=int, default=10,
                        help='Step size for duration thresholds. Default is 10.')
    parser.add_argument('--correlation_threshold', type=float, default=0,
                        help='Correlation threshold for decision boundary analysis. Default is 0.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Whether to print verbose output during analysis.')  

    args = parser.parse_args()

    # save arguments to a dictionary for logging or further processing
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, 'analysis_arguments.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    all_outcomes_bp_events_analysis_pipeline(
        registry_data_path=args.registry_data_path,
        nor_annotated_bp_data_path=args.nor_annotated_bp_data_path,
        correspondance_data_path=args.correspondance_data_path,
        outcome_data_path=args.outcome_data_path,
        output_dir=args.output_dir,
        filter_noradrenaline=args.filter_noradrenaline,
        bp_parameter=args.bp_parameter,
        outcomes=args.outcomes,
        intensity_threshold_range=tuple(args.intensity_threshold_range),
        intensity_threshold_step=args.intensity_threshold_step,
        duration_range=tuple(args.duration_range),
        duration_step=args.duration_step,
        correlation_threshold=args.correlation_threshold,
        verbose=args.verbose
    )
