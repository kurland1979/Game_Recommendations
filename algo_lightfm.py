import pandas as pd
import numpy as np 
import time
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics import roc_auc_score
from merge_files import pipeline_merge
from data_utils import (
    map_indices, split_train_val_test,
    check_cold_start, filter_cold_start,
    precision_at_k,recall_at_k
)
from config import (LIGHT_RANDOM_STATE,LIGHT_NO_COMPONENTS,LIGHT_LOSS,
                    LIGHT_LEARNING_RATE,LIGHT_EPOCHS,LIGHT_USER_ALPHA,LIGHT_ITEM_ALPHA,
                    LIGHT_NUM_THREADS,LIGHT_K)

def build_interactions(train_df):
    """
    Build a LightFM interaction matrix for the training set.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training set with columns ['user_idx', 'item_idx', 'is_recommended'].

    Returns
    -------
    dataset : lightfm.data.Dataset
        Fitted Dataset object containing user and item mappings.
    interactions_train : scipy.sparse.coo_matrix
        User-item interaction matrix for the training set.

    Raises
    ------
    ValueError
        If the provided DataFrame is None or empty.
    """

    if train_df is None or train_df.empty: 
        raise ValueError("One or more DataFrames not provided")
    
    train_df = train_df[['user_idx', 'item_idx', 'is_recommended']].copy()
    
    dataset = Dataset()
    all_users = train_df['user_idx'].unique()
    all_items = train_df['item_idx'].unique()
    dataset.fit(all_users, all_items)


    interactions_train, _ = dataset.build_interactions(zip(train_df['user_idx'],
                                                            train_df['item_idx'],
                                                              train_df['is_recommended']))
   
    return dataset, interactions_train

def train_lightfm(interactions_train):
    """
    Train a LightFM model on the training interactions.

    Parameters
    ----------
    interactions_train : scipy.sparse.coo_matrix
        User-item interaction matrix for the training set.

    Returns
    -------
    algo : LightFM
        The trained LightFM model.
    train_time : float
        Training runtime in seconds.
    """

    algo = LightFM(no_components=LIGHT_NO_COMPONENTS,loss=LIGHT_LOSS,
                   learning_rate=LIGHT_LEARNING_RATE,
                   item_alpha=LIGHT_ITEM_ALPHA,
                   user_alpha=LIGHT_USER_ALPHA,
                   random_state=LIGHT_RANDOM_STATE)
    
    start_time = time.time()
    algo.fit(interactions_train,
             epochs=LIGHT_EPOCHS,
             num_threads=LIGHT_NUM_THREADS,
             )
    train_time = time.time() - start_time

    return algo,train_time

def evaluate(val_df, algo, k=LIGHT_K):
    """
    Evaluate a LightFM model on a validation or test split.

    Parameters
    ----------
    val_df : pandas.DataFrame
        DataFrame containing columns: 'user_idx', 'item_idx', 'is_recommended'.
    algo : LightFM
        Trained LightFM model.
    k : int, optional (default=LIGHT_K)
        Number of top-ranked items to consider for Precision@K and Recall@K.

    Returns
    -------
    results : dict
        Dictionary with the following keys:
        - 'avg_prec' (float): Average Precision@K across all users.
        - 'avg_rec' (float): Average Recall@K across all users.
        - 'auc' (float): Area under the ROC curve for all predictions.
    """
    if val_df is None or val_df.empty: 
        raise ValueError("One or more DataFrames not provided")

    users = list(val_df['user_idx'].unique())
    recalls,precisions = [],[]
    all_y_true,all_y_score = [],[]

    for user in users:
        user_val = val_df[val_df['user_idx'] == user]
        if user_val.empty:
            continue
        
        items = user_val['item_idx'].values
        y_true = user_val['is_recommended'].values

        user_ids = np.repeat(user, len(items))
        y_score = algo.predict(user_ids, items)

        precisions.append(precision_at_k(y_true, y_score, k))
        recalls.append(recall_at_k(y_true, y_score, k))

        all_y_true.append(y_true)
        all_y_score.append(y_score)

    avg_prec = np.mean(precisions)
    avg_rec = np.mean(recalls)

    all_y_true = np.concatenate(all_y_true)
    all_y_score = np.concatenate(all_y_score)
    auc = roc_auc_score(all_y_true, all_y_score)

    results = {
            'avg_prec':avg_prec,
            'avg_rec':avg_rec,
            'auc':auc
            }
    return results

def pipeline_lightfm(verbose=False):
    """
    Full pipeline to train and evaluate a LightFM recommendation model.

    The pipeline performs the following steps:
    1. Merge and preprocess the raw data.
    2. Sort by date and remove duplicate user-item interactions.
    3. Split the data into train, validation, and test sets.
    4. Map users and items to continuous indices.
    5. Check and filter cold-start users and items from validation and test sets.
    6. Build LightFM interaction matrices for train, validation, and test sets.
    7. Train a LightFM model on the training set.
    8. Evaluate the model on validation and test sets using Precision@K, Recall@K, and AUC.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If True, prints shapes and progress information at each pipeline step.

    Returns
    -------
    results_val : dict
        Validation results containing average Precision@K, Recall@K, and AUC.
    results_test : dict
        Test results containing average Precision@K, Recall@K, and AUC.
    """

    df = pipeline_merge()
    if verbose:
        print("Merged DF shape:", df.shape)

    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["user_id", "app_id"], keep="last")
    if verboss:
        print("After drop_duplicates:", df.shape)

    train_df, val_df, test_df = split_train_val_test(df,verbose=False)
   
    train_df, val_df, test_df = map_indices(train_df, val_df, test_df)
    cold_start_users_val, cold_start_items_val, cold_start_users_test, cold_start_items_test = check_cold_start(train_df,
                                                                                                    val_df,
                                                                                                    test_df,
                                                                                                    verbose=False)

    val_df, test_df = filter_cold_start(val_df, test_df,
                        cold_start_users_val, cold_start_items_val,
                        cold_start_users_test, cold_start_items_test,verbose=False)

    dataset, interactions_train = build_interactions(train_df)
    if verbose:
        print("Interactions train:", interactions_train.shape)
        
    algo,train_time = train_lightfm(interactions_train)
    results_val = evaluate(val_df,algo,k=LIGHT_K)
    results_test = evaluate(test_df,algo,k=LIGHT_K)

    return results_val,results_test,train_time
    




