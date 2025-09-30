import pandas as pd
from pandas.errors import DataError
from get_files import pipeline_files


games_df,recoms_df,users_df = pipeline_files()
dataframes = [(games_df,'Games'), (recoms_df,'Recommendations'), (users_df,'Users')]

def check_dataframe(dataframes, verboss=False):
    """
    Runs general checks on a list of DataFrames.

    Parameters
    ----------
    dataframes : list of tuple
        List of (DataFrame, name) tuples to be checked.
    verboss : bool, optional (default=False)
        If True, prints shape, missing values, duplicates, 
        descriptive statistics, and dtypes for each DataFrame.

    Raises
    ------
    DataError
        If no dataframes are provided.

    """

    if not dataframes:
        raise DataError("No dataframes provided")
    try:
        if verboss:
            for df,name in dataframes:
                print(f"--- Checks for {name} ---")
                print("Shape:", df.shape)
                print("Missing values:\n", df.isna().sum())
                print("Duplicates:", df.duplicated().sum())
                print("Describe", df.describe(include='all'))
                print("Dtypes:\n", df.dtypes)
                print("="*40)
    except Exception as e:
        print(f'Error: {e}')

def check_games_file(games_df, verboss=False):
    """
    Performs quality checks on the games dataset.

    Parameters
    ----------
    games_df : pandas.DataFrame
        DataFrame containing the games data.
    verboss : bool, optional (default=False)
        If True, prints counts, unique values, ranges, 
        and distributions of key columns.

    Raises
    ------
    ValueError
        If the DataFrame is empty or not provided.
    """

    if games_df is None or games_df.empty:
        raise ValueError("Games DataFrame is not loaded or is empty")
    try:
        if verboss:
            print("=== Games Data Checks ===")
            print("Unique app_id:", games_df['app_id'].nunique())
            print("Missing titles:", games_df['title'].isna().sum())
            print("Unique titles:", games_df['title'].nunique())
            print("Date range:", games_df['date_release'].min(), "to", games_df['date_release'].max())

            for col in ['win', 'mac', 'linux', 'steam_deck']:
                print(f"Unique values in {col}:", games_df[col].unique())
                print("Rating range:", games_df['rating'].min(), "to", games_df['rating'].max())
                print("Positive ratio range:", games_df['positive_ratio'].min(), "to", games_df['positive_ratio'].max())
                print("User reviews range:", games_df['user_reviews'].min(), "to", games_df['user_reviews'].max())
                print("Price final min/max:", games_df['price_final'].min(), games_df['price_final'].max())
                print("Price original min/max:", games_df['price_original'].min(), games_df['price_original'].max())
                print("Discount range:", games_df['discount'].min(), "to", games_df['discount'].max())

    except Exception as e:
        print(f'Error: {e}')

def check_recommend_file(recoms_df, verboss=False):
    """
    Performs quality checks on the recommendations dataset.

    Parameters
    ----------
    recoms_df : pandas.DataFrame
        DataFrame containing the recommendations data.
    verboss : bool, optional (default=False)
        If True, prints counts, ranges, distributions, 
        and missing values of key columns.

    Raises
    ------
    ValueError
        If the DataFrame is empty or not provided.
    """

    if recoms_df is None or recoms_df.empty:
        raise ValueError("Recommendations DataFrame is not loaded or is empty")
        
    try:
        if verboss:
            print("=== Recommend Data Checks ===")
            print("Unique review_id:", recoms_df['review_id'].nunique())
            print("Missing user_id:", recoms_df['user_id'].isna().sum())
            print("Missing app_id:", recoms_df['app_id'].isna().sum())
            print("Date range:", recoms_df['date'].min(), "to", recoms_df['date'].max())
            print("Unique values in is_recommended:", recoms_df['is_recommended'].unique())
            print("Distribution:\n", recoms_df['is_recommended'].value_counts(normalize=True))
            print("Hours range:", recoms_df['hours'].min(), "to", recoms_df['hours'].max())
            print("Helpful range:", recoms_df['helpful'].min(), "to", recoms_df['helpful'].max())
            print("Funny range:", recoms_df['funny'].min(), "to", recoms_df['funny'].max())

    except Exception as e:
        print(f'Error: {e}')

def check_users_file(users_df, verboss=False):
    """
    Performs quality checks on the users dataset.

    Parameters
    ----------
    users_df : pandas.DataFrame
        DataFrame containing the users data.
    verboss : bool, optional (default=False)
        If True, prints unique user counts, ranges for 
        products and reviews, and checks for invalid cases 
        (reviews > products).

    Raises
    ------
    ValueError
        If the DataFrame is empty or not provided.
    """

    if users_df is None or users_df.empty:
        raise ValueError("Users DataFrame is not loaded or is empty")
    
    try:
        if verboss:
            print("=== Users Data Checks ===")
            print("Unique user_id:", users_df['user_id'].nunique())
            print("Products range:", users_df['products'].min(), "to", users_df['products'].max())
            print("Reviews range:", users_df['reviews'].min(), "to", users_df['reviews'].max())
            invalid = (users_df['reviews'] > users_df['products']).sum()
            print("Users with more reviews than products:", invalid)
    except Exception as e:
        print(f'Error: {e}')

def pipeline_checks():
    """
    Runs a full pipeline of quality checks across all datasets.

    This function executes:
        - General DataFrame checks
        - Games dataset checks
        - Recommendations dataset checks
        - Users dataset checks

    Returns
    -------
    None
    """

    check_dataframe(dataframes,verboss=False)
    check_games_file(games_df,verboss=False)
    check_recommend_file(recoms_df,verboss=False)
    check_users_file(users_df,verboss=True)

    


