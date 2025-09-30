import pandas as pd 
import numpy as np 
from merge_files import pipeline_merge
from sklearn.metrics import roc_auc_score
from data_utils import split_train_val_test,precision_at_k,recall_at_k
from config import K


def global_base(train_df):
    """
    Compute the global mean of the `is_recommended` column.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Input DataFrame containing at least the `is_recommended` column.

    Returns
    -------
    float
        The global mean of `is_recommended` across all rows.

    Raises
    ------
    ValueError
        If the input DataFrame is None or empty.
    """

    if train_df is None or train_df.empty:
        raise ValueError("One or more DataFrames not provided")
    try:
        global_mean = np.mean(train_df['is_recommended'])
    except Exception as e:
        print('ERROR:',e)
    
    return global_mean

def user_base(train_df):
    """
    Compute the mean recommendation score for each user.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Input DataFrame containing `user_id` and `is_recommended` columns.

    Returns
    -------
    pandas.Series
        A Series indexed by `user_id`, where each value is the mean
        of `is_recommended` for that user.

    Raises
    ------
    ValueError
        If the input DataFrame is None or empty.
    """

    if train_df is None or train_df.empty:
        raise ValueError("One or more DataFrames not provided")
    try:
        user_mean = train_df.groupby('user_id')['is_recommended'].mean()
    except Exception as e:
        print('ERROR:',e)

    return user_mean

def item_base(train_df):
    """
    Compute the mean recommendation score for each item (app).

    Parameters
    ----------
    train_df : pandas.DataFrame
        Input DataFrame containing `app_id` and `is_recommended` columns.

    Returns
    -------
    pandas.Series
        A Series indexed by `app_id`, where each value is the mean
        of `is_recommended` for that item.

    Raises
    ------
    ValueError
        If the input DataFrame is None or empty.
    """

    if train_df is None or train_df.empty:
        raise ValueError("One or more DataFrames not provided")
    try:
        item_mean = train_df.groupby('app_id')['is_recommended'].mean()
    except Exception as e:
        print('ERROR:',e)

    return item_mean

def predict(val_df, global_mean, user_mean, item_mean):
    """
    Generate baseline predictions for users and items.

    Predictions are based on the pre-computed user mean and item mean.
    Missing values (cold-start users/items) are filled with the global mean.

    Parameters
    ----------
    val_df : pandas.DataFrame
        DataFrame containing at least `user_id` and `app_id`.
    global_mean : float
        Global mean of `is_recommended` used as a fallback.
    user_mean : pandas.Series
        Series indexed by `user_id`, containing mean scores per user.
    item_mean : pandas.Series
        Series indexed by `app_id`, containing mean scores per item.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added columns:
        - `pred_user` : baseline prediction based on user mean.
        - `pred_item` : baseline prediction based on item mean.
    """

    val_df['pred_user'] = val_df['user_id'].map(user_mean)
    val_df['pred_user'] = val_df['pred_user'].fillna(global_mean)

    val_df['pred_item'] = val_df['app_id'].map(item_mean)
    val_df['pred_item'] = val_df['pred_item'].fillna(global_mean)
    val_df['pred_global'] = global_mean

    return val_df

def evaluate(val_df, k=K):
    """
    Evaluate baseline predictions using Precision@K, Recall@K, and AUC.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Must contain:
        - is_recommended : ground truth labels (0/1)
        - pred_user : predictions from user mean
        - pred_item : predictions from item mean
        - pred_global : predictions from global mean
    global_mean : float
        Global mean of is_recommended used as a fallback.
    k : int, default=10
        Number of top recommendations to use for Precision@K and Recall@K.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - precision_user, recall_user, auc_user
        - precision_item, recall_item, auc_item
        - precision_global, recall_global, auc_global
    """

    y_true = val_df['is_recommended'].values

    # ===== User =====
    prec_user = precision_at_k(y_true, val_df['pred_user'].values, k)
    rec_user = recall_at_k(y_true, val_df['pred_user'].values, k)
    auc_user = roc_auc_score(y_true, val_df['pred_user'].values)

    # ===== Item =====
    prec_item = precision_at_k(y_true, val_df['pred_item'].values, k)
    rec_item = recall_at_k(y_true, val_df['pred_item'].values, k)
    auc_item = roc_auc_score(y_true, val_df['pred_item'].values)

    # ===== Global =====
    prec_global = precision_at_k(y_true, val_df['pred_global'].values, k)
    rec_global = recall_at_k(y_true, val_df['pred_global'].values, k)
    auc_global = roc_auc_score(y_true, val_df['pred_global'].values)

    results_base = {
        "precision_user": prec_user, "recall_user": rec_user, "auc_user": auc_user,
        "precision_item": prec_item, "recall_item": rec_item, "auc_item": auc_item,
        "precision_global": prec_global, "recall_global": rec_global, "auc_global": auc_global
    }

    return results_base


def pipeline_baseline_mean():
    """
    Run the complete baseline (mean-based) pipeline.

    Steps
    -----
    1. Load and preprocess the dataset with `pipeline_merge`.
    2. Compute global, user, and item means.
    3. Generate predictions using user/item/global means.
    4. Evaluate predictions using Precision@K, Recall@K, and AUC.
    5. Measure runtime for the entire pipeline.

    Returns
    -------
    results_base : dict
        Dictionary of evaluation metrics for user, item, and global baselines.
    train_time : float
        Runtime in seconds for executing the baseline pipeline.
    """
    import time
    start_time = time.time()

    # 1. Load data
    df = pipeline_merge()
    train_df, val_df,test_df = split_train_val_test(df,verboss=False)
    train_df = train_df[['user_id', 'app_id', 'is_recommended']].copy()

    # 2. Compute means
    global_mean = global_base(train_df)
    user_mean = user_base(train_df)
    item_mean = item_base(train_df)

    # 3. Generate predictions
    val_df = predict(val_df, global_mean, user_mean, item_mean)
    test_df = predict(test_df, global_mean, user_mean, item_mean)

    # 4. Evaluate
    results_base_val = evaluate(val_df)
    results_base_test = evaluate(test_df)

    # 5. Runtime
    train_time = time.time() - start_time

    return results_base_val,results_base_test, train_time













