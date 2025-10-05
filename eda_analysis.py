import pandas as pd
import numpy as np
from merge_files import pipeline_merge
import matplotlib.pyplot as plt


def eda_games(df, verbose=False):
    """
    Performs exploratory data analysis (EDA) on the games dataset.

    Groups games by release year and computes the number of unique apps,
    average positive ratio, and average user reviews.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing games data with a 'date_release' column.
    verbose : bool, optional (default=False)
        If True, prints descriptive statistics and summary results.

    Returns
    -------
    results : pandas.DataFrame
        Aggregated statistics per release year.
    """

    if df is None or df.empty:
        raise ValueError("DataFrame is not loaded or is empty")
    try:
        results = df.groupby(df['date_release'].dt.year).agg({
                                                'app_id': 'nunique',
                                                'positive_ratio':'mean',
                                                    'user_reviews':'mean'
                                                }).round(2)
        if verbose:
            print('Num_App: ',df['app_id'].nunique())
            print ('\nRange Min/Max For Date Release From: ',df['date_release'].dt.year.min(), "To", df['date_release'].dt.year.max())
            print('\nResult Per Year:\n', results)
        return results
            
    except Exception as e:
        print('ERROR:', e)

def games_per_year_plot(results):
    """
    Plots the number of games released per year.

    Parameters
    ----------
    results : pandas.DataFrame
        Aggregated results from `eda_games`, indexed by release year.

    Saves
    -----
    "games_per_year.png" : bar chart of games released per year.
    """

    x = results.index.astype(int)
    y = results['app_id']
    plt.bar(x, y)
    plt.xlabel("Release Year")
    plt.ylabel("Number of Games")
    plt.title("Games Released per Year")
    plt.xticks(x, x.astype(str))
    plt.xticks(rotation=90)
    plt.savefig("games_per_year.png")
    plt.close()

def positive_per_year_plot(results):
    """
    Plots the average positive ratio per release year.

    Parameters
    ----------
    results : pandas.DataFrame
        Aggregated results from `eda_games`, indexed by release year.

    Saves
    -----
    "positive_per_year.png" : bar chart of positive ratios per year.
    """

    x = results.index.astype(int)
    y = results['positive_ratio']
    plt.bar(x, y)
    plt.xlabel("Release Year")
    plt.ylabel("Number of Positive")
    plt.title("Positive Released per Year")
    plt.xticks(x, x.astype(str))
    plt.xticks(rotation=90)
    plt.savefig("positive_per_year.png")
    plt.close()

def eda_recommendations(df, verbose=False):
    """
    Performs exploratory data analysis (EDA) on the recommendations dataset.

    Groups recommendations by hours played (binned) and calculates:
    - Number of unique users
    - Number of reviews
    - Number of apps
    - Mean recommendation rate

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing recommendations data with 'hours' and 'is_recommended'.
    verbose : bool, optional (default=False)
        If True, prints distributions and aggregated statistics.

    Returns
    -------
    results_per_hours : pandas.DataFrame
        Aggregated statistics per hours bin.
    """

    if df is None or df.empty:
        raise ValueError("DataFrame is not loaded or is empty")
    try:
        bins = [0, 5, 20, 50, 90, float('inf')]
        labels = ["0-5", "5-20", "20-50", "50-90", "90+"]

        df['hours_bin'] = pd.cut(df['hours'], bins=bins, labels=labels, right=False)
        results_per_hours = df.groupby('hours_bin', observed=False).agg({
                'user_id': 'nunique',
                'review_id': 'nunique',
                'app_id': 'nunique',
                'is_recommended': 'mean'   
            }).round(2)
        
        if verbose:
            print('\nCount Recommended:\n',df['is_recommended'].value_counts(normalize=True))
            print(results_per_hours)
        return results_per_hours

    except Exception as e:
        print('ERROR:', e)

def eda_recommendations_plot(results_per_hours):
    """
    Plots recommendation rate per hours bin.

    Parameters
    ----------
    results_per_hours : pandas.DataFrame
        Aggregated statistics from `eda_recommendations`.

    Saves
    -----
    "recommended_per_hours.png" : bar chart of recommendation rates by hours.
    """

    x = results_per_hours.index.astype(str).tolist()  
    y = results_per_hours['is_recommended'].values   
    
    plt.figure(figsize=(8,5))
    plt.bar(x, y)
    plt.xlabel("Hours Bin")
    plt.ylabel("Recommendation Rate")
    plt.title("Recommendation Rate per Hours Bin")
    plt.ylim(0,1)  
    plt.xticks(rotation=45)
    plt.savefig("recommended_per_hours.png")
    plt.close()
  
def eda_users(df, verbose=False):
    """
    Performs exploratory data analysis (EDA) on the users dataset.

    Splits 'reviews' and 'products' into quantile bins, 
    and computes mean recommendation rate per bin.
    Also checks for invalid cases where reviews > products.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing user data with 'reviews', 'products', and 'is_recommended'.
    verbose : bool, optional (default=False)
        If True, prints descriptive statistics, ratio of invalid users,
        and aggregated results.

    Returns
    -------
    None
    """

    if df is None or df.empty:
        raise ValueError("DataFrame is not loaded or is empty")
    
    df['reviews_bin'] = pd.qcut(df['reviews'],q=4, duplicates='drop')
    df['products_bin'] = pd.qcut(df['products'],q=4, duplicates='drop')

    results_per_review = df.groupby('reviews_bin')['is_recommended'].mean().round(2)
    results_per_product= df.groupby('products_bin')['is_recommended'].mean().round(2)
                                         
    try:
        if verbose:
            print('\nDescribe_Products:\n',df['products'].describe())
            print('\nDescribe_Reviews:\n',df['reviews'].describe())
            invalid_mask = df['reviews'] > df['products']
            print('\nRatio Between Reviews To Products: ',invalid_mask.sum() )
            total_users = df['user_id'].nunique()
            invalid_users = df.loc[invalid_mask, 'user_id'].nunique()
            ratio_invalid = invalid_users / total_users
            print(f"Invalid users ratio: {ratio_invalid:.2%}")
            print(results_per_review)
            print(results_per_product)
    except Exception as e:
        print('ERROR:', e)


def pipeline_eda():
    """
    Runs the full exploratory data analysis (EDA) pipeline.

    Steps:
        1. Loads and merges datasets.
        2. Runs games EDA and plots yearly results.
        3. Runs recommendations EDA and plots hours bins.
        4. Runs users EDA with quantile-based analysis.

    Returns
    -------
    None
    """

    df = pipeline_merge()
    results = eda_games(df,verbose=False)
    games_per_year_plot(results)
    positive_per_year_plot(results)
    results_per_hours = eda_recommendations(df,verbose=False)
    eda_recommendations_plot(results_per_hours)
    eda_users(df,verbose=False)




