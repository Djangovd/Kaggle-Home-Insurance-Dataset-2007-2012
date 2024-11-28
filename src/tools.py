import os
import sys
import re
import json
from types import NoneType
import joblib

from re import X

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import xgboost as xgb

from scipy.stats import randint

import shap

### Auxiliary functions
def dates_to_ages(df: pd.DataFrame, date_cols: dict, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Convert dates to ages in years.

    Args:
        df (pd.DataFrame): input DataFrame
        date_cols (dict): column names of dates to convert to ages
        end_date (pd.Timestamp): end date to calculate age from

    Returns:
        pd.DataFrame: dataframe with ages in years
    """
    for date_col, age_col in date_cols.items():
        df[age_col] = (end_date - pd.to_datetime(df[date_col])).dt.days // 365
    return df


def feature_filtering(df: pd.DataFrame, threshold: float = 0.75)-> pd.DataFrame:
    """
    Filter features based on correlation threshold.

    Args:
        df (pd.DataFrame): dataframe to filter
        threshold (float, optional): Threshold for correlations. Defaults to 0.75.

    Returns:
        pd.DataFrame: absolute correlation matrix
    """

    corr_df = df.corr()
    # set diagonal to 0.
    for idx, row in corr_df.iterrows():
        row[idx] = 0.0

    corr_df = corr_df.loc[:, (corr_df > threshold).any(axis=0)]

    corr_df_T = corr_df.T
    corr_df = corr_df_T.loc[:, (corr_df_T > threshold).any(axis=0)]

    return abs(corr_df)



def select_features(df: pd.DataFrame, feature_cols: NoneType | list = None, target_cols: NoneType | list = None)-> tuple:
    """
    Select features and targets from a DataFrame.

    Args:
        df (pd.DataFrame): input DataFrame
        feature_cols (NoneType | list, optional): columns to be used as (input) features. Defaults to None.
        target_cols (NoneType | list, optional): columnt to be used as (target) labels. Defaults to None.

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        tuple: X, y dataset
    """
    if target_cols is not None:
        y = df.loc[:, target_cols].values
    else:
        raise Exception(f"Please provide valid targets")

    if feature_cols is not None:
        X = df.loc[:, feature_cols].values
    else:
        raise Exception(f"Please provide a list of valid features")
    
    return X, y


def apply_pca(X_train: list|np.array, X_test: list|np.array=None, X_val: list|np.array=None, n_components: int|float = 4, pca_cols_prefix: str = "pc_", save: bool=True)-> tuple:
    """
    Apply PCA to the dataset.

    Args:
        X_train (list | np.array): training input data
        X_test (list | np.array, optional): test input data. Defaults to None.
        X_val (list | np.array, optional): validation input data. Defaults to None.
        n_components (int | float, optional): number of PCA components or the accumulated correlation threshold. Defaults to 4.
        pca_cols_prefix (str, optional): prefix for PCA component columns . Defaults to "pc_".
        save (bool, optional): to save or not. Defaults to True.

    Returns:
        tuple: PCA transformed datasets
    """
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(X_train)
    #create columns
    cols = list()

    if n_components < 1.0:
        n_components = pca.n_components_
        
    print(f"Number of principal components: {n_components}")

    for i in range(n_components):
        cols.append(pca_cols_prefix+str(i))
        
    pca_df = pd.DataFrame(data = x_pca, columns = cols)
    
    if X_test is not None:
        X_test_pca = pca.transform(X_test)
    else:
        X_test_pca = None

    if X_val is not None:
        X_val_pca = pca.transform(X_val)
    else:
        X_val_pca = None

    if save:
        joblib.dump(pca, "../models/pca_model.pkl")
        joblib.dump(pca_df, "../data/pca_df.pkl")
        print("PCA model and DataFrame saved.")

    return x_pca, X_test_pca, X_val_pca, pca_df  
