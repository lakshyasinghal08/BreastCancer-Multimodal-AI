"""Label Consistency and Diagnostic Taxonomy Harmonization Engine.

Audits diagnostic ground truth across radiology, pathology, and clinical tables,
maps raw labels to canonical classes (Benign, Malignant, Normal, Unknown, Other),
and evaluates class prevalence and imbalance.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict

import pandas as pd

logger = logging.getLogger("BreastCancer-Multimodal-AI")

# Canonical medical oncology label mapping dictionary
RAW_TO_CANONICAL = {
    # Benign variations
    "b": "Benign",
    "benign": "Benign",
    "benign_without_callback": "Benign",
    "negative": "Benign",
    
    # Malignant variations
    "m": "Malignant",
    "malignant": "Malignant",
    "cancer": "Malignant",
    "positive": "Malignant",
    "idc": "Malignant",
    "infiltrating ductal carcinoma": "Malignant",
    "infiltrating lobular carcinoma": "Malignant",
    "medullary carcinoma": "Malignant",
    "mucinous carcinoma": "Malignant",
    "metaplastic carcinoma": "Malignant",
    
    # Normal variations
    "normal": "Normal",
    "healthy": "Normal",
    "control": "Normal",
    
    # Survival / Status proxies
    "deceased": "Malignant",
    "dead": "Malignant",
    "living": "Other"
}


def map_raw_label_to_canonical(raw_val: Any) -> str:
    """Maps a raw diagnosis/pathology string to canonical diagnostic taxonomy."""
    if pd.isnull(raw_val):
        return "Missing"
    s = str(raw_val).lower().strip()
    if not s or s in ["none", "nan", "null"]:
        return "Missing"

    # Exact dictionary lookup
    if s in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[s]

    # Substring rule matching
    if any(k in s for k in ["malignant", "cancer", "carcinoma", "invasive", "dead"]):
        return "Malignant"
    elif any(k in s for k in ["benign", "negative"]):
        return "Benign"
    elif "normal" in s:
        return "Normal"
    
    return "Unknown"


def audit_label_consistency(
    df_patient_registry: pd.DataFrame,
    clinical_audits: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Evaluates label distributions, mapping consistency, and imbalance across datasets.

    Args:
        df_patient_registry: Patient registry DataFrame.
        clinical_audits: Dictionary of tabular clinical validation summaries.
        logger: Logger instance.

    Returns:
        Structured label audit dictionary containing mapping table, class counts, and imbalance ratios.
    """
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    start_time = time.perf_counter()
    logger.info("Executing Diagnostic Label Consistency and Taxonomy Audit...")

    raw_label_records: List[Dict[str, Any]] = []
    canonical_counts: Counter[str] = Counter()
    dataset_class_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    # 1. Audit labels from Patient Registry
    for _, row in df_patient_registry.iterrows():
        raw_val = row["Diagnosis"]
        canon = map_raw_label_to_canonical(raw_val)
        canonical_counts[canon] += 1
        ds_source = str(row["Dataset Source"])
        dataset_class_counts[ds_source][canon] += 1
        
        raw_label_records.append({
            "Raw Label": str(raw_val),
            "Canonical Class": canon,
            "Source": ds_source
        })

    # Build unique mapping table
    df_raw_all = pd.DataFrame(raw_label_records)
    if not df_raw_all.empty:
        mapping_table = df_raw_all.groupby(["Raw Label", "Canonical Class", "Source"]).size().reset_index(name="Patient Count")
    else:
        mapping_table = pd.DataFrame(columns=["Raw Label", "Canonical Class", "Source", "Patient Count"])

    # 2. Class Imbalance Analysis
    # Benchmark ratio between Malignant and Benign
    n_mal = canonical_counts["Malignant"]
    n_ben = canonical_counts["Benign"]
    n_norm = canonical_counts["Normal"]
    n_unk = canonical_counts["Unknown"]
    n_miss = canonical_counts["Missing"]

    imbalance_ratio = round(max(n_mal, n_ben) / max(1, min(n_mal, n_ben)), 2) if min(n_mal, n_ben) > 0 else 0.0

    class_frequency_df = pd.DataFrame([
        {"Class": "Malignant", "Count": n_mal, "Prevalence Pct": round(n_mal / max(1, len(df_raw_all)) * 100, 2)},
        {"Class": "Benign", "Count": n_ben, "Prevalence Pct": round(n_ben / max(1, len(df_raw_all)) * 100, 2)},
        {"Class": "Normal", "Count": n_norm, "Prevalence Pct": round(n_norm / max(1, len(df_raw_all)) * 100, 2)},
        {"Class": "Unknown", "Count": n_unk, "Prevalence Pct": round(n_unk / max(1, len(df_raw_all)) * 100, 2)},
        {"Class": "Missing", "Count": n_miss, "Prevalence Pct": round(n_miss / max(1, len(df_raw_all)) * 100, 2)},
    ])

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Label audit completed in {elapsed:.2f}s: "
        f"Malignant={n_mal:,}, Benign={n_ben:,}, Normal={n_norm:,}, "
        f"Unknown={n_unk:,}, Missing={n_miss:,}. Imbalance Ratio: {imbalance_ratio}:1."
    )

    return {
        "mapping_table": mapping_table,
        "class_frequency": class_frequency_df,
        "canonical_counts": dict(canonical_counts),
        "dataset_class_breakdown": {k: dict(v) for k, v in dataset_class_counts.items()},
        "imbalance_ratio": imbalance_ratio,
        "audit_runtime_seconds": round(elapsed, 2)
    }
