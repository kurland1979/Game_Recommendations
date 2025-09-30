import pandas as pd
import numpy as np
from pandas.errors import DataError
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from config import RANDOM_STATE,K

def index_map(df):
    """
    Maps user_id and app_id to continuous integer indices for matrix-based models.
    """
    if df is None or df.empty:
        raise DataError("One or more DataFrames not provided")
    try:
        df = df.copy()

        user2idx = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
        idx2user = {idx: user_id for user_id, idx in user2idx.items()}

        item2idx = {item_id: idx for idx, item_id in enumerate(df['app_id'].unique())}
        idx2item = {idx: item_id for item_id, idx in item2idx.items()}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['app_id'].map(item2idx)

        return df, user2idx, idx2user, item2idx, idx2item
    except Exception as e:
        print('ERROR: ', e)

def add_categorical_features(df):
    """
    Add categorical (string) versions of numeric-mapped columns
    for models like LightFM that expect labels.
    """
    rating_map_back = {
        0: 'Mixed',
        1: 'Mostly Positive',
        2: 'Very Positive',
        3: 'Overwhelmingly Positive'
    }
    if 'rating' in df.columns:
        df['rating_cat'] = df['rating'].map(rating_map_back)

    return df


def split_train_val_test(df, method="time",verboss=False):
    """
    Splits dataset into train/val/test.
    method : str
        "random" (default) → stratified random split
        "time" → split by chronological order of reviews (date)
    """

    if method == "random":
        train_val_df, test_df = train_test_split(df, test_size=0.15, 
                                             random_state=RANDOM_STATE,
                                             stratify=df['is_recommended'])
    
        train_df, val_df = train_test_split(train_val_df, test_size=0.176,
                                         random_state=RANDOM_STATE,
                                         stratify=train_val_df['is_recommended'])

    elif method == "time":
        df = df.sort_values("date")

        total = len(df['date'])
        end_train = int(total * 0.70)
        end_val = int(total * 0.85)
        
        train_df = df.iloc[:end_train]
        val_df = df.iloc[end_train:end_val]
        test_df = df.iloc[end_val:]

    else:
        raise ValueError("ERROR: method not identified")
    if verboss:
        print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    return train_df, val_df, test_df

def bins(train_df, verboss=False):
    """
    Apply binning transformations to numeric features for model compatibility.

    This function discretizes continuous columns in the training DataFrame into 
    categorical bins, making them usable as features in LightFM. 
    Specifically:
    - 'price_final' → mapped into fixed price ranges.
    - 'positive_ratio' → mapped into fixed quality score ranges.
    - 'reviews' → divided into quantile-based bins (equal-sized groups).
    - 'products' → divided into quantile-based bins (equal-sized groups).

    Args:
        train_df (pd.DataFrame): Training set containing numeric columns
            ['price_final', 'positive_ratio', 'reviews', 'products'].
        verboss (bool, optional): If True, prints detailed distribution info
            (value counts and ranges for each binned column). Defaults to False.

    Returns:
        tuple:
            - train_df (pd.DataFrame): Updated DataFrame with new categorical 
              columns ['price_bin', 'positive_ratio_bin', 'reviews_bin', 'products_bin'].
            - prod_bins (np.ndarray): Bin edges for 'products', for consistent use 
              in validation/test sets.
            - rev_bins (np.ndarray): Bin edges for 'reviews', for consistent use 
              in validation/test sets.
            - bins_price (list): Bin edges for 'price_final'.
            - labels_price (list): Labels used for price bins.
            - positive_bins (list): Bin edges for 'positive_ratio'.
            - positive_labels (list): Labels used for positive ratio bins.
    """

    bins_price = [0, 4, 20, 29, 40, np.inf]
    labels_price = ["0-4", "5-20", "21-29", "30-40", "40+"]

    train_df['price_bin'] = pd.cut(
        train_df['price_final'], 
        bins=bins_price, 
        labels=labels_price, 
        include_lowest=True
    )
    positive_bins = [0, 60, 80, 90, 95, 100]
    positive_labels = ["0-60", "60-80", "80-90", "90-95", "95-100"]

    train_df['positive_ratio_bin'] = pd.cut(
        train_df['positive_ratio'],
        bins=positive_bins,
        labels=positive_labels,
        include_lowest=True
    )

    train_df['reviews_bin'],rev_bins = pd.qcut(
            train_df['reviews'],q=5, duplicates='drop',retbins=True)

    train_df['products_bin'], prod_bins = pd.qcut(
            train_df['products'], q=5, duplicates='drop', retbins=True
    )
    
    if verboss:
        print("Value Counts Products: ",train_df['products_bin'].value_counts())
        print("Value Counts Reviews: ",train_df['reviews_bin'].value_counts())
        print("Value Counts Price: ", train_df['price_bin'].value_counts())
        print("Value Counts Positive: ", train_df['positive_ratio_bin'].value_counts())
       
        print("Range Products: ", train_df['products_bin'].cat.categories)
        print("Range Reviews: ", train_df['reviews_bin'].cat.categories)
        print("Range Price: ", train_df['price_bin'].cat.categories)
        print("Range Positive: ", train_df['positive_ratio_bin'].cat.categories)
        

    return train_df,prod_bins, rev_bins,bins_price,labels_price,positive_bins,positive_labels
                        

def bins_apply(val_df, test_df, prod_bins, rev_bins, bins_price, labels_price,
               positive_bins, positive_labels, verboss=False):
    """
    Apply consistent binning to validation and test sets.

    This function ensures that the validation and test DataFrames use the same
    binning scheme as the training set. The bin edges (from `bins`) are reused 
    so that categories remain consistent across splits. It applies binning to:
    - 'reviews' using quantile bins from training.
    - 'products' using quantile bins from training.
    - 'price_final' using fixed bins and labels.
    - 'positive_ratio' using fixed bins and labels.

    Args:
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
        prod_bins (np.ndarray): Bin edges for 'products' (from training).
        rev_bins (np.ndarray): Bin edges for 'reviews' (from training).
        bins_price (list): Bin edges for 'price_final'.
        labels_price (list): Labels for price bins.
        positive_bins (list): Bin edges for 'positive_ratio'.
        positive_labels (list): Labels for positive ratio bins.
        verboss (bool, optional): If True, prints counts of missing values in
            each binned column. Defaults to False.

    Returns:
        tuple:
            - val_df (pd.DataFrame): Updated validation DataFrame with 
              ['reviews_bin', 'products_bin', 'price_bin', 'positive_ratio_bin'].
            - test_df (pd.DataFrame): Updated test DataFrame with the same bins.
    """

    
    val_df['reviews_bin'] = pd.cut(val_df['reviews'], bins=rev_bins, include_lowest=True)
    test_df['reviews_bin'] = pd.cut(test_df['reviews'], bins=rev_bins, include_lowest=True)
    
    val_df['products_bin'] = pd.cut(val_df['products'], bins=prod_bins, include_lowest=True)
    test_df['products_bin'] = pd.cut(test_df['products'], bins=prod_bins, include_lowest=True)

    val_df['price_bin'] = pd.cut(val_df['price_final'], bins=bins_price, labels=labels_price, include_lowest=True)
    test_df['price_bin'] = pd.cut(test_df['price_final'], bins=bins_price, labels=labels_price, include_lowest=True)

    val_df['positive_ratio_bin'] = pd.cut(val_df['positive_ratio'],bins=positive_bins,labels=positive_labels,include_lowest=True)
    test_df['positive_ratio_bin'] = pd.cut(test_df['positive_ratio'],bins=positive_bins,labels=positive_labels,include_lowest=True)

    if verboss:
        print("Miss Values Products Val: ",val_df['products_bin'].isna().sum())
        print("Miss Values Products Test: ",test_df['products_bin'].isna().sum())

        print("Miss Values Reviews Val: ",val_df['reviews_bin'].isna().sum())
        print("Miss Values Reviews Test: ",test_df['reviews_bin'].isna().sum())

        print("Miss Values Price Val: ", val_df['price_bin'].isna().sum())
        print("Miss Values Price Test: ", test_df['price_bin'].isna().sum())

        print("Miss Values Positive Val: ", val_df['positive_ratio_bin'].isna().sum())
        print("Miss Values Positive Test: ", test_df['positive_ratio_bin'].isna().sum())

    return val_df, test_df

def map_indices(train_df, val_df, test_df):
    """
    Map user and item IDs to consecutive integer indices based on the training set.

    This function creates mappings (dictionaries) from original user and item IDs 
    to numeric indices, using only the training set as reference. The same mappings 
    are then applied to the validation and test sets to ensure consistency.
    Any rows in validation or test sets that reference unseen users/items are dropped.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training set containing columns ['user_id', 'app_id'].
    val_df : pandas.DataFrame
        Validation set containing columns ['user_id', 'app_id'].
    test_df : pandas.DataFrame
        Test set containing columns ['user_id', 'app_id'].

    Returns
    -------
    train_df : pandas.DataFrame
        Training set with added columns ['user_idx', 'item_idx'].
    val_df : pandas.DataFrame
        Validation set with added columns ['user_idx', 'item_idx'], 
        filtered to include only users/items present in the training set.
    test_df : pandas.DataFrame
        Test set with added columns ['user_idx', 'item_idx'], 
        filtered to include only users/items present in the training set.
    """

    user_map = {uid: idx for idx, uid in enumerate(train_df['user_id'].unique())}
    item_map = {iid: idx for idx, iid in enumerate(train_df['app_id'].unique())}

    train_df['user_idx'] = train_df['user_id'].map(user_map)
    train_df['item_idx'] = train_df['app_id'].map(item_map)

    val_df['user_idx'] = val_df['user_id'].map(user_map)
    val_df['item_idx'] = val_df['app_id'].map(item_map)

    test_df['user_idx'] = test_df['user_id'].map(user_map)
    test_df['item_idx'] = test_df['app_id'].map(item_map)

    val_df = val_df.dropna(subset=['user_idx', 'item_idx']).copy()
    test_df = test_df.dropna(subset=['user_idx', 'item_idx']).copy()

    return train_df, val_df, test_df


def check_cold_start(train_df, val_df, test_df, verboss=False):
    """
    Identifies cold-start users and items in val and test sets.
    """
    users_train = set(train_df['user_idx'].unique())
    users_val = set(val_df['user_idx'].unique())
    users_test = set(test_df['user_idx'].unique())
    items_train = set(train_df['item_idx'].unique())
    items_val = set(val_df['item_idx'].unique())
    items_test = set(test_df['item_idx'].unique())

    cold_start_users_val = users_val - users_train
    cold_start_items_val = items_val - items_train
    cold_start_users_test = users_test - users_train
    cold_start_items_test = items_test - items_train

    if verboss:
        print("Cold start users val:", len(cold_start_users_val))
        print("Cold start items val:", len(cold_start_items_val))
        print("Cold start users test:", len(cold_start_users_test))
        print("Cold start items test:", len(cold_start_items_test))

    return cold_start_users_val, cold_start_items_val, cold_start_users_test, cold_start_items_test

def filter_cold_start(val_df, test_df,
                      cold_start_users_val, cold_start_items_val,
                      cold_start_users_test, cold_start_items_test,verboss=False):
    """
    Removes cold-start users and items from the validation and test sets.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Validation dataset containing user/item indices.
    test_df : pandas.DataFrame
        Test dataset containing user/item indices.
    cold_start_users_val : set
        Users in the validation set not present in the training set.
    cold_start_items_val : set
        Items in the validation set not present in the training set.
    cold_start_users_test : set
        Users in the test set not present in the training set.
    cold_start_items_test : set
        Items in the test set not present in the training set.

    Returns
    -------
    val_df : pandas.DataFrame
        Validation dataset without cold-start users and items.
    test_df : pandas.DataFrame
        Test dataset without cold-start users and items.
    """

    val_df = val_df[
        ~val_df['user_idx'].isin(cold_start_users_val) &
        ~val_df['item_idx'].isin(cold_start_items_val)
    ]

    test_df = test_df[
        ~test_df['user_idx'].isin(cold_start_users_test) &
        ~test_df['item_idx'].isin(cold_start_items_test)
    ]
    if verboss:
        print("Val after filter:", val_df.shape, "Test after filter:", test_df.shape)

    return val_df, test_df

def precision_at_k(y_true, y_score, k=K):
    """
    Compute Precision@K for a ranked list of predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_score : array-like
        Predicted scores or relevance values for each item.
    k : int, default=K
        Number of top-ranked items to consider.

    Returns
    -------
    float
        Precision@K score, i.e., the proportion of relevant items
        among the top-K predicted items.
    """
    idx = np.argsort(y_score)[::-1][:k]
    return np.mean(y_true[idx])


def recall_at_k(y_true, y_score, k=K):
    """
    Compute Recall@K for a ranked list of predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_score : array-like
        Predicted scores or relevance values for each item.
    k : int, default=K
        Number of top-ranked items to consider.

    Returns
    -------
    float
        Recall@K score, i.e., the proportion of relevant items
        retrieved in the top-K predictions out of all relevant items.
    """
    if np.sum(y_true) == 0:
        return 0.0 
    idx = np.argsort(y_score)[::-1][:k]
    return np.sum(y_true[idx]) / np.sum(y_true)

