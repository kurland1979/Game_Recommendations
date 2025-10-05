import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from algo_lightgbm import  prepare_data
from merge_files import pipeline_merge
from sklearn.metrics import precision_recall_curve, roc_curve,auc,precision_score,recall_score,f1_score
from data_utils import split_train_val_test

def compute_roc_metrics(algo, X_val, y_val):
    """
    Compute and plot the ROC curve with AUC for the given model on validation data.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model used for prediction.
    X_val : pandas.DataFrame
        Validation feature set.
    y_val : pandas.Series
        True labels for the validation set.

    Returns
    -------
    tuple
        fpr : numpy.ndarray
            False Positive Rates for different thresholds.
        tpr : numpy.ndarray
            True Positive Rates for different thresholds.
        roc_auc : float
            Area Under the Curve (AUC) score.
    """

    predicted = algo.predict(X_val, num_iteration=algo.best_iteration)
    fpr, tpr, thresholds = roc_curve(y_val, predicted)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    try:
        plt.show()
    except:
        pass
    plt.close()


def pr_curve(algo, X_val, y_val):
    """
    Compute and plot the Precision-Recall (PR) curve with AUC for the given model.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model used for prediction.
    X_val : pandas.DataFrame
        Validation feature set.
    y_val : pandas.Series
        True labels for the validation set.

    Returns
    -------
    tuple
        precision : numpy.ndarray
            Precision values for different thresholds.
        recall : numpy.ndarray
            Recall values for different thresholds.
        pr_auc : float
            Area Under the Precision-Recall Curve (AUC).
    """

    predicted = algo.predict(X_val, num_iteration=algo.best_iteration)
    precision, recall, thresholds = precision_recall_curve(y_val, predicted)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curve.png')
    try:
        plt.show()
    except:
        pass
    plt.close()

def analyze_thresholds(algo, X_val, y_val):
    """
    Analyze model performance across different decision thresholds.

    This function evaluates the trade-off between precision, recall, 
    and F1-score by iterating over a range of thresholds (0 to 1). 
    It applies each threshold to the model's predicted probabilities 
    and computes the corresponding metrics.

    Parameters
    ----------
    algo : lightgbm.Booster
        Trained LightGBM model.
    X_val : pandas.DataFrame
        Validation feature set.
    y_val : pandas.Series
        True labels for the validation set.

    Returns
    -------
    df_thresholds : pandas.DataFrame
        A DataFrame containing threshold values and their corresponding 
        precision, recall, and F1-scores. Sorted by F1-score in 
        descending order to highlight the most balanced thresholds.
    """

    predicted = algo.predict(X_val, num_iteration=algo.best_iteration)
    fpr, tpr, thresholds = roc_curve(y_val, predicted)

    thresholds_to_check = np.linspace(0, 1, 100)
    precisions,recalls,fs = [],[],[]

    for t in thresholds_to_check:
        preds_binary = (predicted >= t).astype(int)
        p = precision_score(y_val, preds_binary, zero_division=0)
        r = recall_score(y_val, preds_binary, zero_division=0)
        f = f1_score(y_val, preds_binary, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        fs.append(f)

    df_thresholds = pd.DataFrame(list(zip(thresholds_to_check, precisions, recalls, fs)),
                             columns=["threshold", "precision", "recall", "f1"])

    df_thresholds = df_thresholds.sort_values("f1", ascending=False).reset_index(drop=True)

    return df_thresholds

def pipeline_thresholds(verbose=False):
    """
    Run threshold analysis pipeline on the trained LightGBM model.

    This function loads the trained LightGBM model, evaluates it on 
    the validation set, and produces diagnostic plots (ROC curve and 
    Precision-Recall curve). It also computes performance metrics 
    (precision, recall, F1) across different thresholds.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If True, prints the top 10 thresholds with the best F1-scores.

    Returns
    -------
    df_thresholds : pandas.DataFrame
        DataFrame containing threshold, precision, recall, and F1-scores,
        sorted by F1 in descending order.
    """

    df = pipeline_merge()
    train_df,val_df,test_df = split_train_val_test(df,verboss=False)
    X_train,y_train,X_val,y_val,X_test,y_test = prepare_data(train_df,val_df,test_df)
    algo = lgb.Booster(model_file="lightgbm_model.txt")
    compute_roc_metrics(algo, X_val, y_val)
    pr_curve(algo, X_val, y_val)
    df_thresholds = analyze_thresholds(algo, X_val, y_val)
    if verbose:
        print(df_thresholds.head(10))


pipeline_thresholds(verbose=False)
