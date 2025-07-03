import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_scatter_multiple_models(file_paths, model_names, colors):

    plt.figure(figsize=(12, 8))

    for i, file_path in enumerate(file_paths):
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

        gt_columns = [col for col in df.columns if "GT" in col]
        error_columns = [col for col in df.columns if "error" in col]

        gt_values = pd.concat([df[col] for col in gt_columns], ignore_index=True)
        error_values = pd.concat([df[col] for col in error_columns], ignore_index=True)

        data = pd.DataFrame({"Ground Truth": gt_values, "Error": error_values})
        data = data[(data["Ground Truth"] > 0) & (data["Error"] <= 1)]

        sns.scatterplot(data=data, x="Ground Truth", y="Error", label=model_names[i],
                        alpha=0.3, s=20, color=colors[i])

        sns.regplot(data=data, x="Ground Truth", y="Error",
                    scatter=False, color=colors[i], lowess=True)

    plt.ylim(0, 0.2)
    plt.title("Prediction Error vs Ground Truth Depth for Multiple Models")
    plt.xlabel("Ground Truth Depth")
    plt.ylabel("Prediction Error (MSE)")
    plt.legend(title="Models")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
