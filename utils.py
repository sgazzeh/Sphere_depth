import pandas as pd
import numpy as np

def load_excel_data(file_path, pred_cols, gt_cols):
    """
    Load prediction and ground truth data from Excel.
    """
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    pred = df[pred_cols].iloc[1:].to_numpy(dtype=np.float32)
    gt = df[gt_cols].iloc[1:].to_numpy(dtype=np.float32)

    return pred, gt

def build_error_dataframe(error_dict):
    """
    Build a DataFrame from a dictionary {(pitch, roll): error}
    """
    records = [(pitch, roll, error) for (pitch, roll), error in error_dict.items()]
    return pd.DataFrame(records, columns=['pitch', 'roll', 'error'])
