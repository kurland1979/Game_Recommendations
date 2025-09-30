from run import print_results,collect_results_pretty,plot_model_comparison
from baseline_mean import pipeline_baseline_mean
from baseline_threshold import pipeline_base_threshold
from baseline_item_item_cf import pipeline_baseline_item_item_cf
from algo_lightfm import pipeline_lightfm
from algo_lightfm_features import pipeline_lightfm_features
from algo_lightgbm import pipeline_lightgbm



def main(verbose=False):
    """
    Run all pipelines, collect summary results, and (optionally) print
    detailed evaluation per model.
    
    Parameters
    ----------
    verbose : bool, default=False
        If True, prints detailed metrics for each model.
    """
   
    df_summary = collect_results_pretty()
    plot_model_comparison(df_summary)
    if verbose:
        
        results_val, results_test, train_time = pipeline_baseline_mean()
        print_results("Baseline (Mean)", results_val, results_test, train_time)

        results_val, results_test, train_time = pipeline_base_threshold()
        print_results("Baseline (Threshold)", results_val, results_test, train_time)

        results_val, results_test, train_time = pipeline_baseline_item_item_cf()
        print_results("Baseline (Item-Item CF)", results_val, results_test, train_time)

        results_val, results_test, train_time = pipeline_lightfm()
        print_results("LightFM", results_val, results_test, train_time)

        results_val, results_test, train_time = pipeline_lightfm_features()
        print_results("LightFM + Features", results_val, results_test, train_time)

        results_val, results_test, train_time = pipeline_lightgbm()
        print_results("LightGBM", results_val, results_test, train_time)


if __name__ == "__main__":
    main(verbose=False)  
 

    

   


