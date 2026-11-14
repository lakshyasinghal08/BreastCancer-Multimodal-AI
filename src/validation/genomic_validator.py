"""Genomic and Transcriptomic Data Integrity Validation Engine.

Audits high-dimensional molecular arrays (microarrays, RNA-seq, CNVs) for
gene identifier consistency, duplicate probes, constant features, variance distribution,
and completeness without performing normalization.
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


def validate_genomic_dataset(
    file_path: Path,
    dataset_name: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Audits a high-dimensional genomic dataset for feature dimensions, probe integrity, and variance.

    Args:
        file_path: Path to the genomic tabular file.
        dataset_name: Name of the dataset.
        logger: Logger instance.

    Returns:
        Structured audit dictionary covering probe count, samples, constant features,
        and variance distributions.
    """
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    start_time = time.perf_counter()
    logger.info(f"Auditing genomic dataset: {file_path.name} in {dataset_name}...")

    # Load file
    sep = "\t" if file_path.suffix in [".tsv", ".txt"] else ","
    # Sample header first to understand orientation (samples as rows or columns)
    df_head = pd.read_csv(file_path, sep=sep, nrows=5)
    
    # Determine orientation
    # In GSE45827: rows are samples (151), columns are probes (54,677)
    # Check if first column has sample identifiers
    n_head_cols = len(df_head.columns)
    
    # Load full dataset or high-dimensional matrix
    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read genomic file {file_path}: {e}")
        return {"error": str(e), "dataset_name": dataset_name}

    n_rows, n_cols = df.shape

    # Identify gene/probe feature columns vs metadata/sample columns
    # Find sample ID column (usually first column or contains sample/id/gsm/tcga)
    sample_id_candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["sample", "id", "gsm", "tcga", "patient"])]
    sample_col = sample_id_candidates[0] if sample_id_candidates else df.columns[0]
    
    # Feature columns: numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()

    n_samples = n_rows
    n_features = len(feature_cols)

    # 1. Gene Identifier Format & Duplicate Features
    gene_ids = [str(c) for c in feature_cols]
    unique_gene_ids = set(gene_ids)
    duplicate_gene_count = len(gene_ids) - len(unique_gene_ids)

    # Probe naming patterns
    probe_patterns = {
        "affymetrix_probes": sum(1 for g in gene_ids if g.endswith("_at")),
        "rna_seq_genes": sum(1 for g in gene_ids if g.startswith("rna_") or g.isupper()),
        "cnv_markers": sum(1 for g in gene_ids if g.startswith("cnv_")),
        "generic_numeric": sum(1 for g in gene_ids if g.isdigit())
    }

    # 2. Missing Value Profiling & Data Completeness
    missing_in_features = int(numeric_df.isnull().sum().sum())
    total_feature_cells = n_samples * max(1, n_features)
    completeness_pct = round((1.0 - (missing_in_features / max(1, total_feature_cells))) * 100, 3)

    # 3. Variance Analysis (Zero & Low Variance Features)
    # Fast vectorized variance calculation
    variances = numeric_df.var(axis=0)
    zero_var_features = variances[variances == 0].index.tolist()
    low_var_features = variances[(variances > 0) & (variances < 0.01)].index.tolist()

    zero_var_count = len(zero_var_features)
    low_var_count = len(low_var_features)

    var_quantiles = {
        "min": float(variances.min()) if len(variances) > 0 else 0.0,
        "q25": float(variances.quantile(0.25)) if len(variances) > 0 else 0.0,
        "median": float(variances.median()) if len(variances) > 0 else 0.0,
        "q75": float(variances.quantile(0.75)) if len(variances) > 0 else 0.0,
        "max": float(variances.max()) if len(variances) > 0 else 0.0
    }

    # 4. Outlier Features (Features with extreme dynamic range > 100 IQR)
    extreme_outlier_genes = []
    if len(variances) > 10:
        v_q75 = var_quantiles["q75"]
        v_iqr = v_q75 - var_quantiles["q25"]
        extreme_thresh = v_q75 + 10 * v_iqr
        extreme_outliers = variances[variances > extreme_thresh]
        extreme_outlier_genes = extreme_outliers.index.tolist()[:20]

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Genomic audit complete for {dataset_name} in {elapsed:.2f}s: "
        f"{n_features:,} features across {n_samples} samples. "
        f"Zero variance features: {zero_var_count}, Low variance: {low_var_count}."
    )

    return {
        "dataset_name": dataset_name,
        "source_filename": file_path.name,
        "sample_count": n_samples,
        "feature_count": n_features,
        "duplicate_feature_count": duplicate_gene_count,
        "completeness_percentage": completeness_pct,
        "total_missing_values": missing_in_features,
        "zero_variance_count": zero_var_count,
        "zero_variance_samples": zero_var_features[:10],
        "low_variance_count": low_var_count,
        "variance_quantiles": var_quantiles,
        "probe_naming_patterns": probe_patterns,
        "extreme_outlier_gene_count": len(extreme_outlier_genes),
        "extreme_outlier_sample": extreme_outlier_genes[:5],
        "audit_runtime_seconds": round(elapsed, 2)
    }
