import json

"""
config.py - Central configuration for the recommendation system project.

This module loads parameters from config.json and exposes them as constants 
for use throughout the pipeline. It covers:
- Paths to input files
- Baseline parameters
- Cross-validation settings
- LightFM and LightGBM hyperparameters
- Recommendation system defaults

Modify config.json to change behavior without editing the code.
"""



with open("/home/kurland/recommendation_system/MY_RECOMMENDATION/config.json", "r") as f:
    config = json.load(f)

# Paths
GAMES_FILE = config["paths"]["games_file"]
RECOMMEND_FILE  = config["paths"]["recommendations_file"]
USERS_FILE = config["paths"]["users_file"]
SAMPLE_SIZE = config["paths"]["sample"]

# Beseline 
K = config['baseline']['k']
THRESHOLD = config['baseline']['threshold']
TOP_N = config['baseline']['top_n']
ITEM_IDX = config['baseline']['item_idx']

# Cross-Validation
N_SPLITS = config["kflod"]["n_splits"]
SHUFFLE = config["kflod"]["shuffle"]
RANDOM_STATE = config["kflod"]["random_state"]

# lightFM no features
LIGHT_NO_COMPONENTS = config["lightfm_no_features"]["no_components"]
LIGHT_LOSS = config["lightfm_no_features"]["loss"]
LIGHT_LEARNING_RATE = config["lightfm_no_features"]["learning_rate"]
LIGHT_EPOCHS = config["lightfm_no_features"]["epochs"]
LIGHT_USER_ALPHA = config["lightfm_no_features"]["user_alpha"]
LIGHT_ITEM_ALPHA = config["lightfm_no_features"]["item_alpha"]
LIGHT_RANDOM_STATE = config["lightfm_no_features"]["random_state"]
LIGHT_NUM_THREADS = config["lightfm_no_features"]["num_threads"]
LIGHT_K = config["lightfm_no_features"]["k"]

# LightFM with features
LIGHT_WITH_NO_COMPONENTS = config["lightfm_with_features"]["no_components"]
LIGHT_WITH_LOSS = config["lightfm_with_features"]["loss"]
LIGHT_WITH_LEARNING_RATE = config["lightfm_with_features"]["learning_rate"]
LIGHT_WITH_EPOCHS = config["lightfm_with_features"]["epochs"]
LIGHT_WITH_USER_ALPHA = config["lightfm_with_features"]["user_alpha"]
LIGHT_WITH_ITEM_ALPHA = config["lightfm_with_features"]["item_alpha"]
LIGHT_WITH_RANDOM_STATE = config["lightfm_with_features"]["random_state"]
LIGHT_WITH_NUM_THREADS = config["lightfm_with_features"]["num_threads"]
LIGHT_WITH_K = config["lightfm_with_features"]["k"]

# LightGBM
LGBM_OBJECTIVE = config['lightgbm']['objective']
LGBM_METRIC = config['lightgbm']['metric']
LGBM_BOOSTING = config['lightgbm']['boosting_type']
LGBM_N_LEAVES = config['lightgbm']['num_leaves']
LGBM_LEARNING = config['lightgbm']['learning_rate']
LGBM_FEATURE = config['lightgbm']['feature_fraction']
LGBM_BAGGING_FRAC = config['lightgbm']['bagging_fraction']
LGBM_BAGGING_FREQ = config['lightgbm']['bagging_freq']
LGBM_VERBOSE = config['lightgbm']['verbose']
LGBM_SEED = config['lightgbm']['seed']
LGBM_NUM_BOOST = config['lightgbm']['num_boost_round']
LGBM_EARLY_STOP = config['lightgbm']['early_stopping_rounds']
LGBM_LOG_EVAL = config['lightgbm']['log_evaluation']
LGBM_K = config['lightgbm']['k']

# Recommendations
REC_THRESHOLD = config['recommendations']['threshold']
REC_TOP_N = config['recommendations']['top_n']
REC_SAMPLE_USER_ID = config['recommendations']['user_id']







