import pandas as pd
import numpy as np
from merge_files import pipeline_merge
from data_utils import split_train_val_test
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif
import matplotlib.pyplot as plt

def prepare_data(train_df):
    """
    Split the training DataFrame into features (X) and target (y).

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset that must include the column `is_recommended`
        along with feature columns.

    Returns
    -------
    X_train : pandas.DataFrame
        Feature set after dropping `is_recommended`, `date`, `title`,
        and `date_release`.
    y_train : pandas.Series
        Target vector containing the values of `is_recommended`.
    """
    train_df['game_age'] = 2025 - train_df['release_year']
    train_df['game_age'] = pd.Timestamp.now().year - train_df['release_year']

    X_train = train_df.drop(columns=['is_recommended','date','title','date_release'])
    y_train = train_df['is_recommended']

    return X_train,y_train

def selectkbest(X_train, y_train):
    """
    Compute feature importance scores using SelectKBest.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature set.
    y_train : pandas.Series
        Target vector (`is_recommended`).

    Returns
    -------
    selector : sklearn.feature_selection.SelectKBest
        Fitted SelectKBest object containing feature importance scores.
    """

    selector = SelectKBest(score_func=mutual_info_classif,k='all')
    selected = selector.fit(X_train,y_train)

    return selected

def run_feature_importance(selected, X_train):
    """
    Extract and organize feature importance scores into a sorted DataFrame.

    Parameters
    ----------
    selected : sklearn.feature_selection.SelectKBest
        Fitted SelectKBest object after training on data.
    X_train : pandas.DataFrame
        Training feature set used to align feature names with scores.

    Returns
    -------
    feature_importance : pandas.DataFrame
        DataFrame with two columns:
        - `feature`: feature name
        - `score`: importance score
        Sorted in descending order by score.
    """

    scores = selected.scores_
    feature_importance = pd.DataFrame({'feature': X_train.columns,'score': scores})
    feature_importance = feature_importance.sort_values(by='score', ascending=False).reset_index(drop=True)

    return feature_importance

def feature_importance_plot(feature_importance):
    """
    Plot and save feature importance scores as a bar chart.

    Parameters
    ----------
    feature_importance : pandas.DataFrame
        DataFrame containing feature names and their corresponding scores.
        Must include columns:
        - `feature`: feature name
        - `score`: importance score

    Saves
    -----
    feature_scores.png : PNG file of the bar chart.

    Notes
    -----
    The plot is also displayed (if possible) and then closed.
    """

    feature_importance.plot(kind='bar')
    plt.title('Feature Importances')
    plt.ylabel('Important Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_scores.png')
    plt.close()

def pipeline_feature_importance(verbose=False):
    """
    Run the full pipeline for feature importance analysis.

    Steps
    -----
    1. Merge and preprocess data with `pipeline_merge`.
    2. Split into train/validation/test using `split_train_val_test`.
    3. Prepare X/y from the training set with `prepare_data`.
    4. Compute feature scores using `selectkbest`.
    5. Organize and plot feature importance with `run_feature_importance` 
       and `feature_importance_plot`.

    Parameters
    ----------
    verbose : bool, optional, default=False
        If True, print the feature importance table.

    Returns
    -------
    None
        The function generates and saves a plot, and optionally prints
        the ranked feature importance.
    """

    df = pipeline_merge()
    train_df, val_df, test_df = split_train_val_test(df,verbose=False)
    X_train,y_train = prepare_data(train_df)
    selected = selectkbest(X_train,y_train)
    feature_importance = run_feature_importance(selected,X_train)
    feature_importance_plot(feature_importance)
    if verboss:
        print("Selected features:")
        print(feature_importance)




pipeline_feature_importance(verboss=False)



