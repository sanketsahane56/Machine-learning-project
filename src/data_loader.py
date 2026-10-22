import os
import pandas as pd
import numpy as np

def check_dataset_files(base_dir="."):
    """
    Checks if required dataset files exist in the project directory.
    Returns a dictionary with existence status and file sizes.
    """
    required_files = {
        "train.csv": os.path.join(base_dir, "train.csv"),
        "test.csv": os.path.join(base_dir, "test.csv"),
        "sample_submission.csv": os.path.join(base_dir, "sample_submission.csv")
    }
    
    results = {}
    for filename, filepath in required_files.items():
        exists = os.path.isfile(filepath)
        size_bytes = os.path.getsize(filepath) if exists else 0
        results[filename] = {
            "path": filepath,
            "exists": exists,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2)
        }
    return results

def load_raw_datasets(base_dir="."):
    """
    Loads train.csv, test.csv, and sample_submission.csv if available.
    Returns (train_df, test_df, sample_sub_df, error_message).
    """
    files_status = check_dataset_files(base_dir)
    
    if not files_status["train.csv"]["exists"]:
        return None, None, None, "train.csv file not found in the project folder!"
    if not files_status["test.csv"]["exists"]:
        return None, None, None, "test.csv file not found in the project folder!"
        
    try:
        train_df = pd.read_csv(files_status["train.csv"]["path"])
        test_df = pd.read_csv(files_status["test.csv"]["path"])
        sample_sub_df = None
        if files_status["sample_submission.csv"]["exists"]:
            sample_sub_df = pd.read_csv(files_status["sample_submission.csv"]["path"])
        return train_df, test_df, sample_sub_df, None
    except Exception as e:
        return None, None, None, f"Error loading dataset: {str(e)}"

def get_dataset_summary(df, name="Dataset"):
    """
    Generates a detailed statistical summary of a given DataFrame.
    """
    if df is None:
        return None
        
    total_rows, total_cols = df.shape
    columns = list(df.columns)
    
    dtypes_dict = {col: str(df[col].dtype) for col in columns}
    missing_dict = {col: int(df[col].isnull().sum()) for col in columns}
    missing_pct_dict = {col: round((df[col].isnull().sum() / total_rows) * 100, 2) for col in columns}
    
    exact_duplicates = int(df.duplicated().sum())
    text_duplicates = int(df.duplicated(subset=['text']).sum()) if 'text' in df.columns else 0
    
    summary = {
        "name": name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": columns,
        "dtypes": dtypes_dict,
        "missing_values": missing_dict,
        "missing_percentages": missing_pct_dict,
        "exact_duplicates": exact_duplicates,
        "text_duplicates": text_duplicates,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }
    
    if 'target' in df.columns:
        target_counts = df['target'].value_counts().to_dict()
        target_ratios = df['target'].value_counts(normalize=True).to_dict()
        summary["target_distribution"] = {
            "counts": {int(k): int(v) for k, v in target_counts.items()},
            "percentages": {int(k): round(float(v) * 100, 2) for k, v in target_ratios.items()}
        }
        
    return summary
