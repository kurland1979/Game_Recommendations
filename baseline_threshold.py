import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from merge_files import pipeline_merge
from config import THRESHOLD,K
from data_utils import index_map, split_train_val_test,precision_at_k,recall_at_k


def evaluate_baseline_threshold(df, threshold, k=K):
    """
    Evaluate a threshold-based baseline on 'products'.

    Prediction rule:
      - products < threshold → 1
      - else → 0
    """
    y_true = df['is_recommended'].values
    y_score = (df['products'] < threshold).astype(int).values

    prec = precision_at_k(y_true, y_score, k)
    rec = recall_at_k(y_true, y_score, k)
    auc = roc_auc_score(y_true, y_score)

    return {"precision": prec, "recall": rec, "auc": auc}

def pipeline_base_threshold():
    """
    Run baseline with threshold on 'products'.

    Steps:
      1. Load data (pipeline_merge).
      2. Map IDs (index_map).
      3. Split into train/val/test.
      4. Evaluate on val and test.
      5. Measure runtime.
    """
    start_time = time.time()

    df = pipeline_merge()
    df, user2idx, idx2user, item2idx, idx2item = index_map(df)
    train_df, val_df, test_df = split_train_val_test(df)

    threshold = train_df['products'].median()

    # Evaluate
    results_val = evaluate_baseline_threshold(val_df, threshold=threshold)
    results_test = evaluate_baseline_threshold(test_df, threshold=threshold)

    train_time = time.time() - start_time

    return results_val, results_test, train_time







