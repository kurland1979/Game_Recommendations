from baseline_mean import pipeline_baseline_mean
from baseline_threshold import pipeline_base_threshold
from baseline_item_item_cf import pipeline_baseline_item_item_cf
from algo_lightfm import pipeline_lightfm
from algo_lightfm_features import pipeline_lightfm_features
from algo_lightgbm import pipeline_lightgbm
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

"""
run.py - Main entry point for running all recommendation pipelines.

This script:
- Executes all baseline and model pipelines (Mean, Threshold, Item-Item CF, LightFM, LightFM + Features, LightGBM).
- Collects validation and test results for each model (Precision, Recall, AUC, Runtime).
- Prints detailed evaluation metrics for each model (via print_results).
- Generates a summary table of all models using tabulate.
- Saves summary results to CSV under outputs/summary_results.csv.
- Produces a comparison bar chart of Validation vs. Test AUC, saved as outputs/models_auc_comparison.png.

Usage:
    python run.py

Notes:
- Designed to work with large-scale Steam dataset (sampled or full).
- Verbose printouts are intended for debugging / in-depth review.
- Summary table and plots provide high-level model comparison.
"""


def print_results(name, results_val, results_test, train_time):
    print(f"\n===== {name} =====")

    # Validation
    print("Validation:")
    if "precision_user" in results_val:  # Baseline (Mean)
        print(f"  User-based    | Precision: {results_val['precision_user']:.4f} | Recall: {results_val['recall_user']:.4f} | AUC: {results_val['auc_user']:.4f}")
        print(f"  Item-based    | Precision: {results_val['precision_item']:.4f} | Recall: {results_val['recall_item']:.4f} | AUC: {results_val['auc_item']:.4f}")
        print(f"  Global-mean   | Precision: {results_val['precision_global']:.4f} | Recall: {results_val['recall_global']:.4f} | AUC: {results_val['auc_global']:.4f}")

    else:  # LightFM, LightGBM, Threshold, CF
        prec_key = [k for k in results_val.keys() if "prec" in k][0]
        rec_key  = [k for k in results_val.keys() if "rec" in k][0]
        auc_key  = [k for k in results_val.keys() if "auc" in k][0]
        print(f"  Precision: {results_val[prec_key]:.4f} | Recall: {results_val[rec_key]:.4f} | AUC: {results_val[auc_key]:.4f}")

    print(f"  Runtime: {train_time:.2f} sec\n")

    # Test
    print("Test:")
    if "precision_user" in results_test:
        print(f"  User-based    | Precision: {results_test['precision_user']:.4f} | Recall: {results_test['recall_user']:.4f} | AUC: {results_test['auc_user']:.4f}")
        print(f"  Item-based    | Precision: {results_test['precision_item']:.4f} | Recall: {results_test['recall_item']:.4f} | AUC: {results_test['auc_item']:.4f}")
        print(f"  Global-mean   | Precision: {results_test['precision_global']:.4f} | Recall: {results_test['recall_global']:.4f} | AUC: {results_test['auc_global']:.4f}")
    else:
        prec_key = [k for k in results_test.keys() if "prec" in k][0]
        rec_key  = [k for k in results_test.keys() if "rec" in k][0]
        auc_key  = [k for k in results_test.keys() if "auc" in k][0]
        print(f"  Precision: {results_test[prec_key]:.4f} | Recall: {results_test[rec_key]:.4f} | AUC: {results_test[auc_key]:.4f}")

    print(f"  Runtime: {train_time:.2f} sec")
    print("=======================================\n")

def collect_results_pretty():
    all_results = []

    # Baseline Mean
    results_val, results_test, train_time = pipeline_baseline_mean()
    all_results.append({
        "Model": "Baseline (Mean)",
        "Val_AUC": results_val.get("auc_user", 0),
        "Test_AUC": results_test.get("auc_user", 0),
        "Precision@K": results_val.get("precision_user", 0),
        "Runtime (s)": round(train_time, 2)
    })

    # Baseline Threshold
    results_val, results_test, train_time = pipeline_base_threshold()
    all_results.append({
        "Model": "Baseline (Threshold)",
        "Val_AUC": results_val.get("auc", 0),
        "Test_AUC": results_test.get("auc", 0),
        "Precision@K": results_val.get("precision", 0),
        "Runtime (s)": round(train_time, 2)
    })

    # Item-Item CF
    results_val, results_test, train_time = pipeline_baseline_item_item_cf()
    all_results.append({
        "Model": "Baseline (Item-Item CF)",
        "Val_AUC": results_val.get("mean_auc", 0),
        "Test_AUC": results_test.get("mean_auc", 0),
        "Precision@K": results_val.get("mean_precision", 0),
        "Runtime (s)": round(train_time, 2)
    })

    # LightFM
    results_val, results_test, train_time = pipeline_lightfm()
    all_results.append({
        "Model": "LightFM",
        "Val_AUC": results_val.get("auc", 0),
        "Test_AUC": results_test.get("auc", 0),
        "Precision@K": results_val.get("avg_prec", 0),
        "Runtime (s)": round(train_time, 2)
    })

    # LightFM + Features
    results_val, results_test, train_time = pipeline_lightfm_features()
    all_results.append({
        "Model": "LightFM + Features",
        "Val_AUC": results_val.get("auc", 0),
        "Test_AUC": results_test.get("auc", 0),
        "Precision@K": results_val.get("avg_prec", 0),
        "Runtime (s)": round(train_time, 2)
    })

    # LightGBM
    results_val, results_test, train_time = pipeline_lightgbm()
    all_results.append({
        "Model": "LightGBM",
        "Val_AUC": results_val.get("auc", 0),
        "Test_AUC": results_test.get("auc", 0),
        "Precision@K": results_val.get("precision", 0),
        "Runtime (s)": round(train_time, 2)
    })

    df_summary = pd.DataFrame(all_results)

    print("\n===== Summary Table =====")
    print(tabulate(df_summary, headers="keys", tablefmt="fancy_grid", showindex=False))

    df_summary.to_csv("summary_results.csv", index=False)

    return df_summary

def plot_model_comparison(df_summary):
    """
    Plot bar chart comparing models by Validation and Test AUC.
    """

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df_summary))

    plt.bar([i - bar_width/2 for i in x], df_summary['Val_AUC'], 
            width=bar_width, label='Validation AUC')
    plt.bar([i + bar_width/2 for i in x], df_summary['Test_AUC'], 
            width=bar_width, label='Test AUC')

    plt.xticks(x, df_summary['Model'], rotation=30, ha="right")
    plt.ylabel("AUC")
    plt.title("Model Comparison by AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models_auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Results saved in outputs/ folder")


