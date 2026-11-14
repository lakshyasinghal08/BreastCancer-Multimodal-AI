"""Duplicate Detection Engine for Multimodal Medical Datasets.

Performs exact hash duplicate detection, perceptual (dHash) near-duplicate
detection, and filename collision analysis without altering or deleting any files.
"""
from __future__ import annotations

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import pandas as pd
from PIL import Image

logger = logging.getLogger("BreastCancer-Multimodal-AI")


def compute_file_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file by reading in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(file_path: Path, hash_size: int = 8) -> Optional[int]:
    """Computes difference hash (dHash) for an image to detect perceptual duplicates.

    Args:
        file_path: Path to the image file.
        hash_size: Grid width (generates hash_size * hash_size bits).

    Returns:
        Integer bit representation of the perceptual difference hash, or None if failed.
    """
    try:
        with Image.open(file_path) as img:
            # Resize to (hash_size + 1, hash_size) grayscale
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(resized.getdata())

            diff = 0
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * (hash_size + 1) + col]
                    right = pixels[row * (hash_size + 1) + col + 1]
                    diff = (diff << 1) | (1 if left > right else 0)
            return diff
    except Exception:
        return None


def hamming_distance(h1: int, h2: int, bit_length: int = 64) -> int:
    """Calculates bitwise Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def detect_duplicates(
    df_inventory: pd.DataFrame,
    sample_images_per_dataset: int = 500,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Orchestrates exact and near-duplicate detection across the entire corpus.

    Args:
        df_inventory: Master inventory DataFrame.
        sample_images_per_dataset: Number of images per dataset to sample for perceptual dHash.
        logger: Logger instance.

    Returns:
        Dictionary containing exact duplicate records, perceptual duplicates,
        and filename collisions.
    """
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    start_time = time.perf_counter()
    logger.info("Starting comprehensive duplicate detection audit...")

    # 1. Filename Collision Audit
    name_to_paths = defaultdict(list)
    for _, row in df_inventory.iterrows():
        p = row["Relative Path"]
        fname = Path(p).name
        name_to_paths[fname].append((row["Dataset Name"], p))

    filename_collisions = []
    for fname, occurrences in name_to_paths.items():
        if len(occurrences) > 1:
            datasets_involved = sorted(list({ds for ds, _ in occurrences}))
            filename_collisions.append({
                "Filename": fname,
                "Occurrence Count": len(occurrences),
                "Datasets Involved": ", ".join(datasets_involved),
                "Sample Paths": "; ".join([p for _, p in occurrences[:3]])
            })
    df_name_collisions = pd.DataFrame(filename_collisions)

    # 2. Exact Hash Duplicate Detection (Candidate filtering by identical file size)
    size_to_rows = defaultdict(list)
    for _, row in df_inventory.iterrows():
        size_to_rows[row["File Size"]].append(row)

    exact_duplicates = []
    # Only test files where file size matches and size > 0
    candidate_sizes = [s for s, rows in size_to_rows.items() if len(rows) > 1 and s > 0]
    logger.info(f"Identified {len(candidate_sizes):,} candidate file-size clusters for hash verification.")

    for s in candidate_sizes:
        rows = size_to_rows[s]
        # Limit candidate check to at most 50 files per size group to maintain audit speed
        hash_to_paths = defaultdict(list)
        for r in rows[:50]:
            fpath = Path(r["Absolute Path"])
            if fpath.exists():
                try:
                    fhash = compute_file_sha256(fpath)
                    hash_to_paths[fhash].append((r["Dataset Name"], r["Relative Path"]))
                except Exception:
                    pass

        for fhash, path_tuples in hash_to_paths.items():
            if len(path_tuples) > 1:
                datasets_involved = sorted(list({ds for ds, _ in path_tuples}))
                exact_duplicates.append({
                    "SHA256 Hash": fhash,
                    "File Size (Bytes)": s,
                    "Duplicate Count": len(path_tuples),
                    "Similarity Score": 1.0,
                    "Datasets Affected": ", ".join(datasets_involved),
                    "Paths": "; ".join([p for _, p in path_tuples[:4]])
                })

    df_exact_duplicates = pd.DataFrame(exact_duplicates)

    # 3. Perceptual (dHash) Near-Duplicate Detection on Sampled Images
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_df = df_inventory[df_inventory["Extension"].isin(image_exts)]
    
    sampled_images = []
    for ds_name in image_df["Dataset Name"].unique():
        ds_subset = image_df[image_df["Dataset Name"] == ds_name]
        sample_n = min(sample_images_per_dataset, len(ds_subset))
        sampled_images.append(ds_subset.sample(n=sample_n, random_state=42))

    if sampled_images:
        df_img_sample = pd.concat(sampled_images, ignore_index=True)
    else:
        df_img_sample = pd.DataFrame()

    perceptual_duplicates = []
    dhashes = []
    for _, row in df_img_sample.iterrows():
        p = Path(row["Absolute Path"])
        dh = compute_dhash(p)
        if dh is not None:
            dhashes.append({
                "path": row["Relative Path"],
                "dataset": row["Dataset Name"],
                "dhash": dh
            })

    # Compare hashes within and across sampled datasets
    for i in range(len(dhashes)):
        for j in range(i + 1, min(i + 50, len(dhashes))):
            dist = hamming_distance(dhashes[i]["dhash"], dhashes[j]["dhash"])
            # Threshold: <= 2 bit differences out of 64 bits = >= 96.8% visual similarity
            if dist <= 2:
                sim_score = round(1.0 - (dist / 64.0), 4)
                perceptual_duplicates.append({
                    "Image 1": dhashes[i]["path"],
                    "Dataset 1": dhashes[i]["dataset"],
                    "Image 2": dhashes[j]["path"],
                    "Dataset 2": dhashes[j]["dataset"],
                    "Hamming Distance": dist,
                    "Similarity Score": sim_score
                })

    df_perceptual_duplicates = pd.DataFrame(perceptual_duplicates)
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Duplicate audit completed in {elapsed:.2f}s: "
        f"{len(df_exact_duplicates)} exact hash duplicate clusters, "
        f"{len(df_perceptual_duplicates)} near-duplicate pairs, "
        f"{len(df_name_collisions)} filename collision clusters."
    )

    return {
        "exact_duplicates": df_exact_duplicates,
        "perceptual_duplicates": df_perceptual_duplicates,
        "filename_collisions": df_name_collisions,
        "metrics": {
            "total_exact_duplicate_clusters": len(df_exact_duplicates),
            "total_near_duplicate_pairs": len(df_perceptual_duplicates),
            "total_filename_collisions": len(df_name_collisions),
            "audit_runtime_seconds": round(elapsed, 2)
        }
    }
