from utils import load_excel_data, build_error_dataframe
from compute_lambda_error import compute_lambda_and_errors, compute_test_errors
from heatmap_plot import plot_error_heatmap
from scatter_plot import plot_scatter_multiple_models

import pandas as pd

def main():

    file_path = ""
    pred_cols_train = []
    gt_cols_train = []

    pred_cols_test = []
    gt_cols_test = []

    # Load data
    pred_train, gt_train = load_excel_data(file_path, pred_cols_train, gt_cols_train)
    pred_test, gt_test = load_excel_data(file_path, pred_cols_test, gt_cols_test)

    # Compute lambdas and errors
    lambdas, errors_train, lambda_avg = compute_lambda_and_errors(pred_train, gt_train)
    errors_test, final_error = compute_test_errors(pred_test, gt_test, lambda_avg)

    # Build error dict for heatmap example
    error_dict = {
    }
    df_heatmap = build_error_dataframe(error_dict)

    # Plot heatmap
    plot_error_heatmap(df_heatmap)

    # Scatter plot across models
    file_paths = [

    ]
    model_names = ['ACDNet', 'SliceNet', 'Depth-Anywhere', 'BiFusev2']
    colors = ['#FFEB3B', '#00BCD4', '#4CAF50', '#F44336']

    plot_scatter_multiple_models(file_paths, model_names, colors)

if __name__ == "__main__":
    main()
