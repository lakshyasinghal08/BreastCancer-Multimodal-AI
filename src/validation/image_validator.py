"""Image Integrity and Quality Validation Engine for Medical AI Datasets.

Validates file readability, corruption, bit-depth, channels, dimensions,
and calculates quality metrics including brightness, contrast, and sharpness.
"""
from __future__ import annotations

import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("BreastCancer-Multimodal-AI")

# Magic bytes dictionary for format identification
MAGIC_BYTES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
    "bmp": b"BM"
}


def check_magic_bytes(header_bytes: bytes) -> str:
    """Identifies the true image format from raw file header magic bytes."""
    if header_bytes.startswith(MAGIC_BYTES["png"]):
        return "png"
    elif header_bytes.startswith(MAGIC_BYTES["jpeg"]):
        return "jpeg"
    elif header_bytes.startswith(MAGIC_BYTES["tiff_le"]) or header_bytes.startswith(MAGIC_BYTES["tiff_be"]):
        return "tiff"
    elif header_bytes.startswith(MAGIC_BYTES["bmp"]):
        return "bmp"
    elif len(header_bytes) >= 132 and header_bytes[128:132] == b"DICM":
        return "dicom"
    return "unknown"


def validate_image_file(file_path: Path) -> Dict[str, Any]:
    """Inspects an individual image file for corruption, header integrity, and dimensions.

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary containing validation flags, dimensions, format, and status.
    """
    res: Dict[str, Any] = {
        "file_path": str(file_path),
        "filename": file_path.name,
        "is_valid": True,
        "is_corrupted": False,
        "is_zero_byte": False,
        "is_unreadable": False,
        "wrong_extension": False,
        "extremely_small": False,
        "extremely_large": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "color_mode": "unknown",
        "format": "unknown",
        "file_size_bytes": 0,
        "error_message": ""
    }

    try:
        stat = file_path.stat()
        res["file_size_bytes"] = stat.st_size

        if stat.st_size == 0:
            res["is_valid"] = False
            res["is_zero_byte"] = True
            res["error_message"] = "File is zero bytes"
            return res

        # Header magic check
        with open(file_path, "rb") as f:
            header = f.read(140)

        detected_fmt = check_magic_bytes(header)
        ext = file_path.suffix.lower().lstrip(".")
        if ext in ["jpg", "jpeg"]:
            ext = "jpeg"
        elif ext in ["tif", "tiff"]:
            ext = "tiff"

        if detected_fmt != "unknown" and ext in ["png", "jpeg", "tiff", "bmp"]:
            if detected_fmt != ext:
                res["wrong_extension"] = True

        # Open and verify image structure
        with Image.open(file_path) as img:
            img.verify()  # Verify file integrity

        # Reopen to read dimensions and mode (verify closes or invalidates img state)
        with Image.open(file_path) as img:
            w, h = img.size
            res["width"] = w
            res["height"] = h
            res["color_mode"] = img.mode
            res["format"] = img.format or detected_fmt
            
            # Channel count
            if img.mode in ["L", "1", "I", "I;16"]:
                res["channels"] = 1
            elif img.mode in ["RGB", "YCbCr"]:
                res["channels"] = 3
            elif img.mode in ["RGBA", "CMYK"]:
                res["channels"] = 4
            else:
                res["channels"] = len(img.getbands())

            if w < 10 or h < 10:
                res["extremely_small"] = True
            if w > 10000 or h > 10000 or stat.st_size > 100 * 1024 * 1024:
                res["extremely_large"] = True

    except Exception as e:
        res["is_valid"] = False
        res["is_corrupted"] = True
        res["is_unreadable"] = True
        res["error_message"] = str(e)

    return res


def compute_image_quality_metrics(file_path: Path) -> Dict[str, float]:
    """Computes empirical quality metrics (brightness, contrast, sharpness, noise).

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary of brightness, contrast, sharpness, and noise estimates.
    """
    metrics = {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "noise": 0.0,
        "aspect_ratio": 1.0
    }
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            metrics["aspect_ratio"] = round(w / max(1, h), 4)

            # Convert to grayscale numpy array for photometric analysis
            gray = np.array(img.convert("L"), dtype=np.float32)

            # Brightness (mean pixel luminance)
            metrics["brightness"] = float(np.mean(gray))

            # Contrast (RMS contrast / standard deviation)
            metrics["contrast"] = float(np.std(gray))

            # Sharpness (Laplacian variance approximation)
            if gray.shape[0] >= 3 and gray.shape[1] >= 3:
                # Discrete 3x3 Laplacian kernel
                lap = (
                    -4 * gray[1:-1, 1:-1]
                    + gray[:-2, 1:-1]
                    + gray[2:, 1:-1]
                    + gray[1:-1, :-2]
                    + gray[1:-1, 2:]
                )
                metrics["sharpness"] = float(np.var(lap))

                # Noise estimation (high frequency difference)
                diff_h = np.diff(gray, axis=1)
                diff_v = np.diff(gray, axis=0)
                metrics["noise"] = float((np.std(diff_h) + np.std(diff_v)) / 2.0)

    except Exception:
        pass

    return metrics
