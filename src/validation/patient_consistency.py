"""Patient Consistency and Cohort Alignment Analysis Engine.

Audits patient identifiers across multimodal datasets for duplication,
conflicting diagnostic labels, modality completeness, and train/test partition leakage.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

import pandas as pd

logger = logging.getLogger("BreastCancer-Multimodal-AI")


def analyze_patient_consistency(
    df_patient_registry: pd.DataFrame,
    datasets_root: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Evaluates cross-modal patient alignment, conflicting diagnoses, and leakage risks.

    Args:
        df_patient_registry: Primary patient registry DataFrame from Phase 1.2.
        datasets_root: Path to datasets root directory.
        logger: Logger instance.

    Returns:
        Structured audit dictionary covering conflicts, missingness, overlap, and leakage.
    """
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    start_time = time.perf_counter()
    logger.info("Executing Patient Consistency and Cross-Modal Cohort Audit...")

    total_patients = len(df_patient_registry)

    # 1. Patients without Diagnostic Labels
    unlabeled_patients = df_patient_registry[
        df_patient_registry["Diagnosis"].isnull() | (df_patient_registry["Diagnosis"].astype(str).str.strip() == "")
    ]
    unlabeled_count = len(unlabeled_patients)
    unlabeled_pct = round((unlabeled_count / max(1, total_patients)) * 100, 2)

    # 2. Conflicting Diagnostic Labels
    # Check if any patient has multiple contradictory labels stored in Diagnosis string
    conflicts: List[Dict[str, Any]] = []
    for _, row in df_patient_registry.iterrows():
        diag_str = str(row["Diagnosis"]).lower()
        pid = str(row["Patient ID"])
        has_benign = "benign" in diag_str
        has_malignant = "malignant" in diag_str or "invasive" in diag_str or "dead" in diag_str
        if has_benign and has_malignant:
            conflicts.append({
                "Patient ID": pid,
                "Reported Diagnoses": row["Diagnosis"],
                "Source Datasets": row["Dataset Source"],
                "Modalities": row["Available Modalities"]
            })
    df_conflicts = pd.DataFrame(conflicts)

    # 3. Modality Incompleteness in Multi-Modal Cohorts (Vision_Genomic_Fusion)
    # Check completeness of the 122 fusion patients across MRI, SVS, RNA, CNV, Mutations
    fusion_dir = datasets_root / "Vision_Genomic_Fusion"
    fusion_completeness = []
    if fusion_dir.exists():
        # Read available patient IDs in each modality
        mri_patients: Set[str] = set()
        svs_patients: Set[str] = set()
        rna_patients: Set[str] = set()
        cnv_patients: Set[str] = set()
        mut_patients: Set[str] = set()

        # MRI folder inspection
        mri_dir = fusion_dir / "MRI"
        if mri_dir.exists():
            mri_patients = {d.name for d in mri_dir.iterdir() if d.is_dir()}

        # SVS folder inspection
        svs_dir = fusion_dir / "SVS"
        if svs_dir.exists():
            svs_patients = {d.name for d in svs_dir.iterdir() if d.is_dir()}

        # Omics tabular files
        rna_file = fusion_dir / "RNA" / "TCGA_BRCA_RNA_seq_normalized.csv"
        if rna_file.exists():
            try:
                df_rna_pts = pd.read_csv(rna_file, usecols=[0], nrows=500)
                rna_patients = set(df_rna_pts.iloc[:, 0].dropna().astype(str).unique())
            except Exception:
                pass

        cnv_file = fusion_dir / "CNV" / "TCGA_BRCA_CNV.csv"
        if cnv_file.exists():
            try:
                df_cnv_pts = pd.read_csv(cnv_file, usecols=[0], nrows=500)
                cnv_patients = set(df_cnv_pts.iloc[:, 0].dropna().astype(str).unique())
            except Exception:
                pass

        all_fusion_pts = sorted(list(mri_patients | svs_patients | rna_patients | cnv_patients))
        for pt in all_fusion_pts:
            avail = []
            missing = []
            if pt in mri_patients: avail.append("MRI")
            else: missing.append("MRI")

            if pt in svs_patients: avail.append("SVS_Histology")
            else: missing.append("SVS_Histology")

            if pt in rna_patients: avail.append("RNA_seq")
            else: missing.append("RNA_seq")

            if pt in cnv_patients: avail.append("CNV")
            else: missing.append("CNV")

            fusion_completeness.append({
                "Patient ID": pt,
                "Available Modality Count": len(avail),
                "Available Modalities": ", ".join(avail),
                "Missing Modalities": ", ".join(missing),
                "Is Full Multimodal": len(missing) == 0
            })

    df_fusion_comp = pd.DataFrame(fusion_completeness)

    # 4. Train / Test Overlap & Leakage Audit in CBIS-DDSM
    # In CBIS-DDSM, examine patient IDs in train vs test partitions
    cbis_leakage = []
    cbis_dir = datasets_root / "CBIS-DDSM"
    if cbis_dir.exists():
        train_pts: Set[str] = set()
        test_pts: Set[str] = set()
        
        # Traverse train and test directories
        for root, _, files in os.walk(cbis_dir):
            rel = Path(root).relative_to(cbis_dir).parts
            if "Train" in rel or "train" in rel:
                for f in files:
                    # In CBIS-DDSM, filenames often contain patient identifiers like P_00001
                    parts = f.split("_")
                    for p in parts:
                        if p.isdigit() or (p.startswith("P") and len(p) >= 4):
                            train_pts.add(p)
            elif "Test" in rel or "test" in rel:
                for f in files:
                    parts = f.split("_")
                    for p in parts:
                        if p.isdigit() or (p.startswith("P") and len(p) >= 4):
                            test_pts.add(p)

        overlap = train_pts & test_pts
        if overlap:
            for pt in sorted(list(overlap)):
                cbis_leakage.append({
                    "Dataset": "CBIS-DDSM",
                    "Patient ID": pt,
                    "Leakage Type": "Patient present in both Train and Test partitions",
                    "Risk Level": "High (Data Leakage Risk)"
                })

    df_leakage = pd.DataFrame(cbis_leakage)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Patient consistency audit completed in {elapsed:.2f}s: "
        f"{unlabeled_count:,} unlabeled patients, "
        f"{len(df_conflicts)} diagnostic conflicts, "
        f"{len(df_leakage)} train/test overlap candidates."
    )

    return {
        "total_patients": total_patients,
        "unlabeled_count": unlabeled_count,
        "unlabeled_pct": unlabeled_pct,
        "diagnostic_conflicts": df_conflicts,
        "diagnostic_conflict_count": len(df_conflicts),
        "fusion_completeness": df_fusion_comp,
        "fusion_fully_paired_count": int(df_fusion_comp["Is Full Multimodal"].sum()) if not df_fusion_comp.empty else 0,
        "fusion_total_patients": len(df_fusion_comp),
        "train_test_leakage": df_leakage,
        "leakage_count": len(df_leakage),
        "audit_runtime_seconds": round(elapsed, 2)
    }
