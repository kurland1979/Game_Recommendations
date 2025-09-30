import pandas as pd
import numpy as np
from algo_lightgbm import  prepare_data
from merge_files import pipeline_merge
from data_utils import split_train_val_test
import lightgbm as lgb
from config import REC_THRESHOLD,REC_TOP_N,REC_SAMPLE_USER_ID

def build_recommendations(algo, X_test, test_df,user_id=REC_SAMPLE_USER_ID,
                          threshold=REC_THRESHOLD, top_n=REC_TOP_N):
    """
    Generate top-N game recommendations for a given user using a trained LightGBM model.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model used for prediction.
    X_test : pandas.DataFrame
        Feature set of the test dataset.
    test_df : pandas.DataFrame
        Test dataset containing 'user_id', 'app_id', and 'title' columns.
    user_id : int, optional
        The ID of the user for whom recommendations are generated (default: REC_SAMPLE_USER_ID).
    threshold : float, optional
        Minimum score threshold to consider a game as recommended (default: REC_THRESHOLD).
    top_n : int, optional
        Number of top recommendations to return (default: REC_TOP_N).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top-N recommended games for the given user,
        including 'app_id', 'title', and prediction score.
    """

    if algo is None:
        raise ValueError("Model not provided")
    
    predicted = algo.predict(X_test,num_iteration=algo.best_iteration)
    test_df = test_df.copy() 
    test_df['score'] = predicted
    user_idx = test_df[test_df['user_id'] == user_id]
    user_idx = user_idx[user_idx['score'] >= threshold]
    user_idx = user_idx.sort_values(by='score', ascending=False)
    top_recs = user_idx[['app_id', 'price_final', 'score']].head(top_n)
    
    return top_recs

def demonstrate_recommendation_system(algo, X_test, test_df):
    """
    Demonstrate the recommendation system by generating sample outputs 
    at different decision thresholds (conservative, balanced, liberal).

    This function does not return values, but prints the number of 
    recommendations found for each threshold scenario.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model used for prediction.
    X_test : pandas.DataFrame
        Feature matrix for the test set.
    test_df : pandas.DataFrame
        Test dataset containing 'user_id' and 'app_id'.
    user_id : int, optional
        The ID of the user to generate recommendations for 
        (default: REC_SAMPLE_USER_ID).
    """


    print("=== Game Recommendation System Demo ===\n")
    
    print("Conservative Recommendations (threshold=0.9):")
    recs_conservative = build_recommendations(algo, X_test, test_df, 
                                             user_id=REC_SAMPLE_USER_ID, 
                                             threshold=0.9, top_n=5)
    print(f"Found {len(recs_conservative)} highly confident recommendations")
    
    print("\nBalanced Recommendations (threshold=0.7):")
    recs_balanced = build_recommendations(algo, X_test, test_df,
                                         user_id=REC_SAMPLE_USER_ID,
                                         threshold=0.7, top_n=5)
    print(f"Found {len(recs_balanced)} recommendations")
    
    print("\nLiberal Recommendations (threshold=0.5):")
    recs_liberal = build_recommendations(algo, X_test, test_df,
                                        user_id=REC_SAMPLE_USER_ID,
                                        threshold=0.5, top_n=5)
    print(f"Found {len(recs_liberal)} recommendations")
    
    


def pipeline_recommendations():
    """
    Run the end-to-end recommendation pipeline using a trained LightGBM model.

    This pipeline:
    - Loads and merges the dataset.
    - Splits data into train/validation/test sets.
    - Prepares features and labels.
    - Loads the trained LightGBM model from file.
    - Generates top-N recommendations for a sample user.
    - Demonstrates recommendations at different thresholds 
      (conservative, balanced, liberal).

    Returns
    -------
    None
        Prints the top recommendations and demo results directly.
    """

    df = pipeline_merge()
    train_df,val_df,test_df = split_train_val_test(df,verboss=False)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df,val_df,test_df)
    algo = lgb.Booster(model_file="lightgbm_model.txt")
    top_recs = build_recommendations(algo, X_test, test_df, user_id=REC_SAMPLE_USER_ID,
                          threshold=REC_THRESHOLD, top_n=REC_TOP_N)
    
    print(top_recs)
    demonstrate_recommendation_system(algo, X_test, test_df,user_id=REC_SAMPLE_USER_ID)

pipeline_recommendations()
    




















