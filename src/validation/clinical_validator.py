"""Clinical & EHR Data Integrity Validation Engine.

Audits tabular clinical and EHR datasets for missing values, impossible values,
duplicate rows, categorical inconsistencies, and statistical outliers without cleaning.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("BreastCancer-Multimodal-AI")


def validate_clinical_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    source_filename: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Audits a clinical or EHR DataFrame for integrity, types, bounds, and anomalies.

    Args:
        df: Clinical DataFrame to validate.
        dataset_name: Name of the dataset containing the table.
        source_filename: Filename of the table.
        logger: Logger instance.

    Returns:
        Structured audit dictionary covering missingness, duplicates, outliers, and bounds.
    """
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    start_time = time.perf_counter()
    n_rows, n_cols = df.shape

    # 1. Missing Value Profiling
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    total_cells = n_rows * n_cols
    overall_missing_pct = round((total_missing / max(1, total_cells)) * 100, 2)
    col_missing_pct = (missing_counts / max(1, n_rows) * 100).round(2).to_dict()

    # 2. Row and Patient Duplication
    duplicate_rows_count = int(df.duplicated().sum())

    # Detect candidate patient ID column
    id_candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["id", "patient", "barcode", "sample"])]
    patient_id_col = id_candidates[0] if id_candidates else None
    duplicate_patient_count = int(df[patient_id_col].duplicated().sum()) if patient_id_col else 0

    # 3. Candidate Diagnosis / Target Columns
    target_candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["diagnosis", "status", "stage", "label", "class"])]
    target_col = target_candidates[0] if target_candidates else None

    missing_diagnosis_count = 0
    unique_diagnoses: List[str] = []
    if target_col:
        missing_diagnosis_count = int(df[target_col].isnull().sum())
        unique_diagnoses = [str(x) for x in df[target_col].dropna().unique()[:20]]

    # 4. Impossible Value Checks
    impossible_values: List[Dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        col_lower = str(col).lower()
        vals = df[col].dropna()
        # Check negative age
        if "age" in col_lower and (vals < 0).any():
            bad_cnt = int((vals < 0).sum())
            impossible_values.append({
                "Column": col,
                "Anomaly": "Negative Age",
                "Violations": bad_cnt,
                "Sample": vals[vals < 0].iloc[0]
            })
        # Check negative physical dimensions (radius, texture, perimeter, area)
        if any(k in col_lower for k in ["radius", "area", "perimeter", "texture", "size"]) and (vals <= 0).any():
            bad_cnt = int((vals <= 0).sum())
            impossible_values.append({
                "Column": col,
                "Anomaly": "Non-Positive Physical Dimension",
                "Violations": bad_cnt,
                "Sample": vals[vals <= 0].iloc[0]
            })

    # 5. Outlier Summary using IQR (Tukey's fences)
    outlier_summary: List[Dict[str, Any]] = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 10:
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = vals[(vals < lower_bound) | (vals > upper_bound)]
            if len(outliers) > 0:
                outlier_summary.append({
                    "Column": col,
                    "Outlier Count": len(outliers),
                    "Outlier Pct": round(len(outliers) / len(vals) * 100, 2),
                    "Lower Bound": round(lower_bound, 4),
                    "Upper Bound": round(upper_bound, 4),
                    "Min Found": round(float(vals.min()), 4),
                    "Max Found": round(float(vals.max()), 4)
                })

    # 6. Data Types Consistency
    dtype_map = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}

    elapsed = time.perf_counter() - start_time
    logger.info(f"Clinical audit completed for {dataset_name}/{source_filename} in {elapsed:.3f}s.")

    return {
        "dataset_name": dataset_name,
        "source_filename": source_filename,
        "total_rows": n_rows,
        "total_columns": n_cols,
        "total_missing_cells": total_missing,
        "overall_missing_pct": overall_missing_pct,
        "column_missing_pct": col_missing_pct,
        "duplicate_rows": duplicate_rows_count,
        "patient_id_column": patient_id_col,
        "duplicate_patients": duplicate_patient_count,
        "target_column": target_col,
        "missing_diagnosis_count": missing_diagnosis_count,
        "unique_diagnoses": unique_diagnoses,
        "impossible_values": impossible_values,
        "outlier_summary": outlier_summary,
        "dtypes": dtype_map,
        "audit_runtime_seconds": round(elapsed, 3)
    }
