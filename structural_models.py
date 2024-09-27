#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint
import sys
sys.path.append('/home/ss2686/03_DICTrank')
from scripts.evaluation_functions import evaluate_classifier, optimize_threshold_j_statistic

# Constants
DIRIL_FILES = ['../data/1083_csv_diri.csv', '../data/317_csv_diri.csv']
RANDOM_STATE = 42

def generate_fingerprints(smiles_list):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)

def load_and_prepare_data(file_paths):
    data = pd.concat([pd.read_csv(file) for file in file_paths])
    X = generate_fingerprints(data['SMILES'])
    y = (data['Nephrotoxicity'] == 'Nephrotoxic').astype(int)
    return data, X, y

def get_model_and_param_dist():
    model = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    param_dist = {
        'max_depth': randint(10, 20),
        'max_features': randint(40, 50),
        'min_samples_leaf': randint(5, 15),
        'min_samples_split': randint(5, 15),
        'n_estimators': [200, 300, 400, 500, 600],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    return model, param_dist

def optimize_model(X_train, y_train):
    model, param_dist = get_model_and_param_dist()
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = HalvingRandomSearchCV(
        model,
        param_dist,
        factor=3,
        cv=inner_cv,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def perform_cross_validation(X_train, y_train, best_model):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sampler = RandomOverSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
    oof_predictions = np.zeros(X_train.shape[0])
    oof_probs = np.zeros(X_train.shape[0])
    cv_scores = []

    for train_idx, valid_idx in inner_cv.split(X_train, y_train):
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_valid_fold, y_valid_fold = X_train[valid_idx], y_train[valid_idx]

        X_resampled, y_resampled = sampler.fit_resample(X_train_fold, y_train_fold)
        best_model.fit(X_resampled, y_resampled)

        oof_predictions[valid_idx] = best_model.predict(X_valid_fold)
        oof_probs[valid_idx] = best_model.predict_proba(X_valid_fold)[:, 1]

        fold_auc = roc_auc_score(y_valid_fold, oof_probs[valid_idx])
        cv_scores.append(fold_auc)

    return oof_probs, cv_scores

def main():
    results = {}
    held_out_results = []

    # Load and prepare data
    diril_data, X, y = load_and_prepare_data(DIRIL_FILES)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Optimize model
    best_model = optimize_model(X_train, y_train)

    # Perform cross-validation
    oof_probs, cv_scores = perform_cross_validation(X_train, y_train, best_model)

    # Random Over-sampling and Threshold Optimization
    sampler = RandomOverSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[('sampler', sampler), ('model', best_model)])
    pipeline.fit(X_train, y_train)

    # Predict using threshold-optimized model
    probs_test = pipeline.predict_proba(X_test)[:, 1]

    # Optimize the threshold using out-of-fold predictions
    best_threshold = optimize_threshold_j_statistic(y_train, oof_probs)
    predictions_test = (probs_test >= best_threshold).astype(int)

    # Collect results
    results['DIRIL_Nephrotoxicity'] = {
        'CV_AUC_mean': np.mean(cv_scores),
        'CV_AUC_std': np.std(cv_scores),
        **evaluate_classifier(y_test, predictions_test, probs_test)
    }

    held_out_data = {
        'Dataset': 'DIRIL',
        'Activity': 'Nephrotoxicity',
        'SMILES': diril_data.loc[X_test.index, 'SMILES'],
        'True_Value': y_test,
        'Prediction': predictions_test,
        'Probability': probs_test,
        'Best_Threshold': best_threshold
    }

    held_out_results.append(pd.DataFrame(held_out_data))

    # Save results
    pd.DataFrame(results).T.to_csv('./structural_model_results_DIRIL.csv')
    pd.DataFrame(held_out_data).to_csv('./structural_model_held_out_test_results_DIRIL.csv', index=False)

if __name__ == "__main__":
    main()
