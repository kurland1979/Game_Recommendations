import pandas as pd
import numpy as np
import os
from config import GAMES_FILE,RECOMMEND_FILE,USERS_FILE,SAMPLE_SIZE 

def preview_games_file(verboss=False):
    """
    Loads a preview of the games dataset.

    Parameters
    ----------
    verboss : bool, optional (default=False)
        If True, prints column names and the first few rows.

    Returns
    -------
    games_df : pandas.DataFrame
        A DataFrame containing a preview of the games file.
    """

    if not os.path.exists(GAMES_FILE):
         raise FileNotFoundError(f"Games file not found at {GAMES_FILE}")
     
    games_df = pd.read_csv(GAMES_FILE, nrows=SAMPLE_SIZE)

    if verboss:
        print("Columns:", games_df.columns.tolist())
        print(games_df.head())
    return games_df

def preview_recommendations_file(verboss=False):
    """
    Loads a preview of the recommendations dataset.

    Parameters
    ----------
    verboss : bool, optional (default=False)
        If True, prints column names and the first few rows.

    Returns
    -------
    recoms_df : pandas.DataFrame
        A DataFrame containing a preview of the recommendations file.
    """

    if not os.path.exists(RECOMMEND_FILE):
        raise FileNotFoundError(f"Recommendations file not found at {RECOMMEND_FILE}")
    recoms_df = pd.read_csv(RECOMMEND_FILE, nrows=SAMPLE_SIZE)
    if verboss:
        print("Columns:", recoms_df.columns.tolist())
        print(recoms_df.head())
    return recoms_df
    
def preview_users_file(verboss=False):
    """
    Loads a preview of the users dataset.

    Parameters
    ----------
    verboss : bool, optional (default=False)
        If True, prints column names and the first few rows.

    Returns
    -------
    users_df : pandas.DataFrame
        A DataFrame containing a preview of the users file.
    """

    if not os.path.exists(USERS_FILE):
        raise FileNotFoundError(f"Users file not found at {USERS_FILE}")
    users_df = pd.read_csv(USERS_FILE, nrows=SAMPLE_SIZE)
    if verboss:
        print("Columns:", users_df.columns.tolist())
        print(users_df.head())
    return users_df
    
def pipeline_files():
    """
    Loads and previews all datasets used in the recommendation system.

    Returns
    -------
    games_df : pandas.DataFrame
        Preview of the games dataset.
    recoms_df : pandas.DataFrame
        Preview of the recommendations dataset.
    users_df : pandas.DataFrame
        Preview of the users dataset.
    """

    games_df = preview_games_file(verboss=False)
    recoms_df = preview_recommendations_file(verboss=False)
    users_df = preview_users_file(verboss=False)
    return games_df,recoms_df,users_df