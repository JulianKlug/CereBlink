import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import io
import getpass
import msoffcrypto
from sklearn.model_selection import KFold

from static_prediction.evaluate_model import evaluate_model

continuous_predictor_variables = [
"Age",
"GCS_admission",
"Aneurysm_diameter"
]

categorical_predictor_variables = [
"Sex",
"Smoker_0no_1yes_2ex",
"Drinker",
"WFNS",
"HH",
"CN_deficit_admission_YN",
"Sedation_on_admission_YN",
"Intubated_on_admission_YN",
"HTN",
"DM",
"Statin",
"ASS",
"Clopidogrel",
"OAC",
"Fisher_Score",
"IVH",
"SDH",
"ICH",
"Multiple_Aneurysms_2unk",
"Nr_of_Aneurysms",
"Aneurysm_Artery_Code",
"Aneurysm_Side",
"Aneurysm_FusiformBlep_YN",
"Aneurysm_Distal_YN",
"Aneurysm_Mycotic_YN",
"Aneurysm_Blister_YN",
"Coiling",
"Complete_Coiling",
"Clipping",
"Complete_Clipping",
"Stenting",
"EVD_YN",
"CVS_YN",
"Nimotop"
]

def cv_eval_dci_prediction_from_static(df: pd.DataFrame, predictor_variables, outcome) -> pd.DataFrame:

    pids = df.pid.unique()

    models = []
    results_df = pd.DataFrame()
    # cross validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(pids):
        model = xgb.XGBClassifier(enable_categorical=True)
        train_pids = pids[train_index]
        test_pids = pids[test_index]

        train_df = df[df.pid.isin(train_pids)]
        test_df = df[df.pid.isin(test_pids)]

        train_x = train_df[predictor_variables]
        train_y = train_df[outcome]

        test_x = test_df[predictor_variables]
        test_y = test_df[outcome]

        model.fit(train_x, train_y)
        fold_result_df = evaluate_model(model, test_x, test_y, outcome, model_config={})

        results_df = pd.concat([results_df, fold_result_df])
        models.append(model)

    return results_df, models


def safe_float_conversion(x):
    try:
        return float(x)
    except:
        return np.nan


def preprocess_sah_registry_data(df: pd.DataFrame, outcome_var, categorical_predictor_variables,
                                 continuous_predictor_variables) -> pd.DataFrame:
    df['pid'] = df['SOS-CENTER-YEAR-NO.']

    df[categorical_predictor_variables] = df[categorical_predictor_variables].astype('category')
    df['Age'] = df['Age'].astype(float).apply(lambda x: np.abs(x))

    for col in continuous_predictor_variables:
        df[col] = df[col].apply(safe_float_conversion)

    df_with_outcome = df[~df[outcome_var].isna()]

    return df_with_outcome


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-O', '--outcome', type=str, required=True)
    args = parser.parse_args()

    password = getpass.getpass()
    decrypted_workbook = io.BytesIO()
    with open(args.input_file_path, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=password)
        office_file.decrypt(decrypted_workbook)

    registry_df = pd.read_excel(decrypted_workbook, sheet_name='DATA')

    df = preprocess_sah_registry_data(registry_df, outcome_var=args.outcome,
                                      categorical_predictor_variables=categorical_predictor_variables,
                                      continuous_predictor_variables=continuous_predictor_variables)

    results_df, models = cv_eval_dci_prediction_from_static(df,
                                                    predictor_variables=continuous_predictor_variables + categorical_predictor_variables,
                                                    outcome=args.outcome)

    output_file_path = os.path.join(args.output_folder, f'{args.outcome}_dci_prediction_results.csv')
    results_df.to_csv(output_file_path, index=False)

    pickle.dump(models, open(os.path.join(args.output_folder, f'{args.outcome}_dci_prediction_models.pkl'), 'wb'))

