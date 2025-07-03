import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_error_heatmap(df, output_file="output/error_heatmap.png"):
    """
    Plot and save heatmap.
    """
    matrix = df.pivot(index='pitch', columns='roll', values='error').sort_index(ascending=False)
    annot = matrix.copy().applymap(lambda x: f"{np.floor(x*100)/100:.2f}" if pd.notnull(x) else "")

    error_min = df['error'].min()
    error_max = df['error'].max()

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=annot, fmt="", cmap="coolwarm",
                cbar_kws={'label': 'Error'}, vmin=error_min, vmax=error_max)
    plt.title("Error Heatmap: Pitch vs Roll")
    plt.xlabel("Roll Angle")
    plt.ylabel("Pitch Angle")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
