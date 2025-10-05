import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from merge_files import pipeline_merge
from data_utils import split_train_val_test,precision_at_k,recall_at_k
from config import (LGBM_OBJECTIVE,LGBM_METRIC,LGBM_BOOSTING,
                    LGBM_N_LEAVES,LGBM_LEARNING,LGBM_FEATURE,
                    LGBM_BAGGING_FRAC,LGBM_BAGGING_FREQ,
                    LGBM_VERBOSE,LGBM_SEED,
                    LGBM_NUM_BOOST,LGBM_EARLY_STOP,
                    LGBM_LOG_EVAL,LGBM_K)

def prepare_data(train_df, val_df, test_df):
    """
    Prepare feature matrices (X) and target vectors (y) for train, validation, and test sets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing both features and the target column `is_recommended`.
    val_df : pandas.DataFrame
        Validation dataset containing both features and the target column `is_recommended`.
    test_df : pandas.DataFrame
        Test dataset containing both features and the target column `is_recommended`.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features with non-relevant columns removed (`is_recommended`, `date`, `title`, `date_release`).
    y_train : pandas.Series
        Training target (`is_recommended`).
    X_val : pandas.DataFrame
        Validation features with the same columns as X_train.
    y_val : pandas.Series
        Validation target (`is_recommended`).
    X_test : pandas.DataFrame
        Test features with the same columns as X_train.
    y_test : pandas.Series
        Test target (`is_recommended`).
    """

    if train_df is None or train_df.empty: 
        raise ValueError("One or more DataFrames not provided")
    
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    
    columns_to_use = train_df.drop(columns=['is_recommended','date','title','date_release']).columns
    X_train = train_df[columns_to_use]
    y_train = train_df['is_recommended']

    X_val = val_df[columns_to_use]
    y_val = val_df['is_recommended']

    X_test = test_df[columns_to_use]
    y_test = test_df['is_recommended']
    
    return X_train,y_train,X_val,y_val,X_test,y_test

def build_dataset(X_train, y_train, X_val, y_val):
    """
    Build LightGBM Dataset objects for training and validation.

    Converts the input feature matrices and labels into LightGBM-specific 
    Dataset objects. These datasets are optimized for efficient training 
    and are required as input for the LightGBM training process.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target vector.
    X_val : pandas.DataFrame or numpy.ndarray
        Validation feature matrix.
    y_val : pandas.Series or numpy.ndarray
        Validation target vector.

    Returns
    -------
    trainset : lightgbm.Dataset
        Dataset object containing training features and labels.
    valset : lightgbm.Dataset
        Dataset object containing validation features and labels, 
        with `trainset` used as a reference.
    """

    trainset = lgb.Dataset(X_train,label=y_train,free_raw_data=False)
    valset = lgb.Dataset(X_val,label=y_val,reference=trainset,free_raw_data=False)
    
    return trainset, valset

def train_algo(trainset, valset):
    """
    Train a LightGBM model using the provided training and validation datasets.

    Configures LightGBM with hyperparameters defined in the configuration 
    file, then trains the model while monitoring validation performance.
    Includes early stopping and periodic logging.

    Parameters
    ----------
    trainset : lightgbm.Dataset
        Training dataset containing features and labels.
    valset : lightgbm.Dataset
        Validation dataset used for monitoring performance and early stopping.

    Returns
    -------
    algo : lightgbm.Booster
        Trained LightGBM model.
    train_time : float
        Total training runtime in seconds.
    """

    params = {
        'objective': LGBM_OBJECTIVE,
        'metric': LGBM_METRIC,
        'boosting_type': LGBM_BOOSTING,
        'num_leaves': LGBM_N_LEAVES,
        'learning_rate': LGBM_LEARNING,
        'feature_fraction': LGBM_FEATURE,
        'bagging_fraction': LGBM_BAGGING_FRAC,
        'bagging_freq': LGBM_BAGGING_FREQ,
        'verbose': LGBM_VERBOSE,
        'seed': LGBM_SEED
        }
    
    start_time = time.time()
    
    algo = lgb.train(params=params,
                     train_set=trainset,
                     num_boost_round=LGBM_NUM_BOOST,
                     valid_sets=[trainset,valset],
                     valid_names=['trainset','valset'],
                     callbacks=[lgb.early_stopping(LGBM_EARLY_STOP),
                                lgb.log_evaluation(LGBM_LOG_EVAL)])

    train_time = time.time() - start_time

    return algo,train_time

def evaluate(algo, X_val, y_val):
    """
    Evaluate a trained LightGBM model on validation data.

    Generates predictions and computes evaluation metrics including 
    Precision@K, Recall@K, and AUC. The value of K is taken from the 
    global configuration.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model.
    X_val : pandas.DataFrame
        Feature set of the validation data.
    y_val : pandas.Series
        True labels of the validation data.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'precision' : float
            Precision@K score.
        - 'recall' : float
            Recall@K score.
        - 'auc' : float
            Area Under the ROC Curve (AUC) score.
    """

    predicted = algo.predict(X_val,num_iteration=algo.best_iteration)
    precision = precision_at_k(y_val,predicted,k=LGBM_K)
    recall = recall_at_k(y_val,predicted,k=LGBM_K)
    auc = roc_auc_score(y_val,predicted)

    results = {'precision':precision,
               'recall':recall,
               'auc':auc}
    
    return results

def pipeline_lightgbm():
    """
    Full training and evaluation pipeline for the LightGBM model.

    This function merges data, prepares train/validation/test splits,
    builds LightGBM datasets, trains the model, evaluates it, and 
    stores app titles for recommendation purposes.

    Returns
    -------
    results_val : dict
        Evaluation metrics (precision, recall, AUC) for validation data.
    results_test : dict
        Evaluation metrics (precision, recall, AUC) for test data.
    train_time : float
        Total training time in seconds.
   
    """

    df = pipeline_merge()
    train_df,val_df,test_df = split_train_val_test(df,verbose=False)
    X_train,y_train,X_val,y_val,X_test,y_test = prepare_data(train_df,val_df,test_df)
    trainset, valset = build_dataset(X_train,y_train,X_val,y_val)
    algo,train_time = train_algo(trainset,valset)
    algo.save_model("lightgbm_model.txt")
    results_val = evaluate(algo,X_val,y_val)
    results_test = evaluate(algo,X_test,y_test)
    
    return results_val,results_test,train_time
















