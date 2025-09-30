import pandas as pd
import numpy as np 
import time
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics import roc_auc_score
from merge_files import pipeline_merge
from data_utils import (
    map_indices,
    split_train_val_test,
    check_cold_start, filter_cold_start,
    bins,bins_apply,
    precision_at_k,recall_at_k
)
from config import (LIGHT_WITH_RANDOM_STATE,LIGHT_WITH_NO_COMPONENTS,LIGHT_WITH_LOSS,
                    LIGHT_WITH_LEARNING_RATE,LIGHT_WITH_EPOCHS,LIGHT_WITH_USER_ALPHA,LIGHT_WITH_ITEM_ALPHA,
                    LIGHT_WITH_NUM_THREADS,LIGHT_WITH_K)

def build_interactions_features(train_df):
    """
    Build interaction matrix along with user and item feature matrices for the LightFM model.

    This function prepares the interaction data and encodes additional user/item features
    such as binned statistics and platform compatibility flags. It fits a LightFM Dataset
    object with all users, items, and their associated features, and outputs the 
    interaction matrix, user features matrix, and item features matrix.

    Args:
        train_df (pd.DataFrame): Training DataFrame containing the following columns:
            - user_idx (int): Encoded user IDs.
            - item_idx (int): Encoded item IDs.
            - products_bin (category): Binned product counts per user.
            - reviews_bin (category): Binned review counts per user.
            - price_bin (category): Binned item price ranges.
            - positive_ratio_bin (category): Binned positive review ratio.
            - win (int): 1 if the game supports Windows, else 0.
            - mac (int): 1 if the game supports MacOS, else 0.
            - linux (int): 1 if the game supports Linux, else 0.
            - steam_deck (int): 1 if the game is compatible with Steam Deck, else 0.
            - is_recommended (int): Target label (1 if recommended, 0 otherwise).

    Returns:
        tuple:
            - interactions_train (scipy.sparse.coo_matrix): Sparse interaction matrix 
              representing user-item interactions.
            - user_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of user features.
            - item_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of item features.

    Raises:
        ValueError: If the provided DataFrame is None or empty.
    """

    if train_df is None or train_df.empty: 
        raise ValueError("One or more DataFrames not provided")
    
    train_df = train_df[['user_idx', 'item_idx','products_bin','reviews_bin',
                         'price_bin','positive_ratio_bin','win','mac','linux',
                         'steam_deck',
                         'is_recommended']].copy()
    
    dataset = Dataset()
    all_users = train_df['user_idx'].unique()
    all_items = train_df['item_idx'].unique()

    products_features = list(train_df['products_bin'].dropna().unique())
    reviews_features = list(train_df['reviews_bin'].dropna().unique())
    users_features = products_features + reviews_features 

    price_features = list(train_df['price_bin'].dropna().unique())
    positive_features = list(train_df['positive_ratio_bin'].dropna().unique())
    
    items_features = price_features + positive_features + ['win', 'mac', 'linux', 'steam_deck']

    dataset.fit(all_users, all_items,user_features=users_features,item_features=items_features)

    user_features = [
    (u, list(set([p, r]))) for u, p, r in zip(train_df['user_idx'],
                                               train_df['products_bin'],
                                                 train_df['reviews_bin'],
                                                 )]
    item_features = [
    (i, list(set(
        [pr, pos] +
        (['win'] if w == 1 else []) +
        (['mac'] if m == 1 else []) +
        (['linux'] if l == 1 else []) +
        (['steam_deck'] if sd == 1 else [])
    )))
    for i, pr, pos, w, m, l, sd in zip(
        train_df['item_idx'],
        train_df['price_bin'],
        train_df['positive_ratio_bin'],
        train_df['win'],
        train_df['mac'],
        train_df['linux'],
        train_df['steam_deck']   
    )]

    interactions_train, weights = dataset.build_interactions(zip(train_df['user_idx'],
                                                            train_df['item_idx'],
                                                              train_df['is_recommended']))
    
    user_features_matrix = dataset.build_user_features(user_features)
    item_features_matrix = dataset.build_item_features(item_features)

    return interactions_train,user_features_matrix,item_features_matrix

def train_lightfm_features(interactions_train, user_features_matrix, item_features_matrix):
    """
    Train a LightFM model with user and item features.

    This function initializes a LightFM model with hyperparameters from the 
    global configuration and trains it on the given interaction matrix along 
    with user and item feature matrices. The function also measures and returns 
    the training runtime.

    Args:
        interactions_train (scipy.sparse.coo_matrix): Sparse interaction matrix 
            representing user-item interactions.
        user_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of user features.
        item_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of item features.

    Returns:
        tuple:
            - algo (LightFM): Trained LightFM model.
            - train_time (float): Training runtime in seconds.
    """

    algo = LightFM(no_components=LIGHT_WITH_NO_COMPONENTS,loss=LIGHT_WITH_LOSS,
                    learning_rate=LIGHT_WITH_LEARNING_RATE,
                    item_alpha=LIGHT_WITH_ITEM_ALPHA,
                    user_alpha=LIGHT_WITH_USER_ALPHA,
                    random_state=LIGHT_WITH_RANDOM_STATE)
    
    start_time = time.time()
    algo.fit(interactions=interactions_train,
             user_features=user_features_matrix,
             item_features=item_features_matrix,
             epochs=LIGHT_WITH_EPOCHS,
             num_threads=LIGHT_WITH_NUM_THREADS)
    
    train_time = time.time() - start_time
    
    return algo,train_time

def evaluate(val_df, algo, user_features_matrix, item_features_matrix, k=LIGHT_WITH_K):
    """
    Evaluate a trained LightFM model on a validation set.

    The function computes Precision@K, Recall@K, and AUC scores for each user 
    in the validation set, then aggregates results into averages and returns them.

    Args:
        val_df (pd.DataFrame): Validation DataFrame containing at least 
            ['user_idx', 'item_idx', 'is_recommended'] columns.
        algo (LightFM): Trained LightFM model to evaluate.
        user_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of user features.
        item_features_matrix (scipy.sparse.csr_matrix): Sparse matrix of item features.
        k (int, optional): Number of top items to consider for Precision@K and 
            Recall@K. Defaults to LIGHT_WITH_K from global config.

    Returns:
        dict: A dictionary with the following keys:
            - 'avg_prec' (float): Average Precision@K across all users.
            - 'avg_rec' (float): Average Recall@K across all users.
            - 'auc' (float): ROC AUC score over all predictions.
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
        y_score = algo.predict(user_ids, items,
                               user_features=user_features_matrix,
                               item_features=item_features_matrix
                               )

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

def pipeline_lightfm_features(verboss=False):
    """
    Full LightFM pipeline with user and item features.

    This function executes the entire recommendation workflow:
    - Merge and clean raw data.
    - Split into train, validation, and test sets.
    - Apply binning transformations on numeric features (reviews, products, price, positive ratio).
    - Map user/item IDs to internal indices.
    - Handle cold-start cases by filtering unseen users/items.
    - Build LightFM interaction matrix, user features, and item features.
    - Train the LightFM model.
    - Evaluate performance on both validation and test sets.

    Args:
        verboss (bool, optional): If True, prints debug information 
            about data shapes, transformations, and matrices. Defaults to False.

    Returns:
        tuple:
            - results_val (dict): Validation metrics with keys 
                {'avg_prec', 'avg_rec', 'auc'}.
            - results_test (dict): Test metrics with the same keys as validation.
            - train_time (float): Model training time in seconds.
    """

    df = pipeline_merge()
    if verboss:
        print("Merged DF shape:", df.shape)
    
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["user_id", "app_id"], keep="last")
    if verboss:
        print("After drop_duplicates:", df.shape)

    train_df, val_df, test_df = split_train_val_test(df,verboss=False)
    
    train_df,prod_bins, rev_bins,bins_price,labels_price,positive_bins,positive_labels = bins(train_df,
                                                                                              verboss=False)
    val_df, test_df = bins_apply(val_df, test_df,prod_bins,rev_bins,bins_price,
                                 labels_price,positive_bins,positive_labels,                              
                                 verboss=False)

    train_df, val_df, test_df = map_indices(train_df, val_df, test_df)
    cold_start_users_val, cold_start_items_val, cold_start_users_test, cold_start_items_test = check_cold_start(train_df,
                                                                                                    val_df,
                                                                                                    test_df,
                                                                                                    verboss=False)

    val_df, test_df = filter_cold_start(val_df, test_df,
                        cold_start_users_val, cold_start_items_val,
                        cold_start_users_test, cold_start_items_test,verboss=False)

    interactions_train,user_features_matrix,item_features_matrix = build_interactions_features(train_df)
    algo,train_time = train_lightfm_features(interactions_train,user_features_matrix,item_features_matrix)
    if verboss:
        print('matrix train:',interactions_train.shape)
        print('matrix users',user_features_matrix.shape)
        print('matrix items',item_features_matrix.shape)

    results_val = evaluate(val_df,algo,user_features_matrix,item_features_matrix, k=LIGHT_WITH_K)
    results_test = evaluate(test_df,algo,user_features_matrix,item_features_matrix, k=LIGHT_WITH_K)

    return results_val, results_test,train_time
