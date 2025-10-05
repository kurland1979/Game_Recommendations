import pandas as pd
from pandas.errors import DataError
from get_files import pipeline_files

def merge_files(games_df, recoms_df, users_df):
    """
    Merges the games, recommendations, and users datasets.

    Parameters
    ----------
    games_df : pandas.DataFrame
        DataFrame containing the games data.
    recoms_df : pandas.DataFrame
        DataFrame containing the recommendations data.
    users_df : pandas.DataFrame
        DataFrame containing the users data.

    Returns
    -------
    df : pandas.DataFrame
        Merged DataFrame with all datasets combined on `app_id` and `user_id`.

    Raises
    ------
    DataError
        If one or more DataFrames are missing.
    """

    if games_df is None or recoms_df is None or users_df is None:
        raise DataError("One or more DataFrames not provided")
    try:
        merge_df = pd.merge(games_df,recoms_df,how='inner',on='app_id')
        df = pd.merge(merge_df,users_df,how='inner',on='user_id')
    except Exception as e:
        print(f'ERROR: {e}')
    return df
    
def map_columns(df):
    """
    Maps and converts column values to numeric format.

    - Converts 'rating' text labels into integers.
    - Casts boolean/binary columns (is_recommended, win, mac, linux, steam_deck) to integers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing merged game, recommendation, and user data.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame with mapped and converted columns.
    """

    map_names = {'Mixed':0, 'Mostly Positive':1, 'Very Positive':2, 'Overwhelmingly Positive':3}
    df['rating'] = df['rating'].map(map_names)
    
    df['is_recommended'] = df['is_recommended'].astype(int)
    df['win'] = df['win'].astype(int)
    df['mac'] = df['mac'].astype(int)
    df['linux'] = df['linux'].astype(int)
    df['steam_deck'] = df['steam_deck'].astype(int)
    
    return df
    
def convert_date(df):
    """
    Converts date columns to datetime format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing `date` and `date_release` columns.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with converted datetime columns.
    """

    df['date'] = pd.to_datetime(df['date'])
    df['date_release'] = pd.to_datetime(df['date_release'])
    df['release_year'] = df['date_release'].dt.year
    
    return df
    
def basic_checks(df, verbose=False):
    """
    Runs basic checks on a DataFrame for validation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check.
    verbose : bool, optional (default=False)
        If True, prints descriptive statistics, shape, info, 
        duplicates count, and missing values.

    Raises
    ------
    DataError
        If the DataFrame is not provided.
    """

    if df is None:
        raise DataError(" DataFrame not provided")
        
    if verbose:
        print('Describe\n',df.describe())
        print('\nShape:', df.shape)
        print('\nInfo:\n',df.info())
        print('\nDuplicated:', df.duplicated().sum())
        print('\nMissing_Values\n',df.isnull().sum())

def pipeline_merge():
    """
    Executes the full preprocessing pipeline.

    Steps:
        1. Loads all datasets.
        2. Merges them into one DataFrame.
        3. Maps categorical columns to numeric.
        4. Converts date columns to datetime.
        5. Runs basic validation checks.

    Returns
    -------
    df : pandas.DataFrame
        Preprocessed and validated dataset ready for modeling.
    """

    games_df,recoms_df,users_df = pipeline_files()
    df = merge_files(games_df,recoms_df,users_df)
    df = map_columns(df)
    df = convert_date(df)
    basic_checks(df,verbose=False)
    
    return df
