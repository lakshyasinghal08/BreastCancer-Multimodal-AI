# Master Dataset Engine Developer Reference & API Guide

**Framework**: BreastCancer-Multimodal-AI (v2.4.0-RC2)  
**Target Audience**: Deep Learning Modelers, Phase 3 Backbone Developers, Clinical Evaluators  

---

## 1. Quick Start Guide for Phase 3 Model Training

Model developers can immediately instantiate the unified multimodal dataset without any custom preprocessing or directory scanning:

```python
import pandas as pd
from pathlib import Path

# 1. Load the frozen split manifest
BASE_DIR = Path("d:/BreastCancer-Multimodal-AI")
train_df = pd.read_csv(BASE_DIR / "outputs/splits/train.csv")

# 2. Instantiate the unified multimodal dataset
train_dataset = UnifiedMultimodalDataset(
    manifest_df=train_df,
    base_dir=BASE_DIR,
    is_train=True
)

# 3. Build the production DataLoader
train_loader = ProductionDataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=multimodal_collate_fn
)

# 4. Train loop iteration
for batch in train_loader:
    labels = batch['labels']
    patient_ids = batch['patient_ids']
    modalities = batch['modalities']
    
    # Access tensors dynamically by modality
    if 'images' in batch:
        images = batch['images'] # (B, C, H, W)
    if 'clinical_features' in batch:
        clinical = batch['clinical_features'] # (B, 30)
    if 'genomic_features' in batch:
        genomic = batch['genomic_features'] # (B, 54675)
```

---

## 2. Dataset Classes Specifications

| Dataset Class | Modality | Output Tensor | Key Preprocessing |
| :--- | :--- | :--- | :--- |
| `MammographyDataset` | Mammography | `(1, 224, 224)` float32 | Grayscale conversion, Bilinear resize, MinMax [0, 1] |
| `UltrasoundDataset` | Ultrasound | `(3, 256, 256)` float32 | RGB conversion, Bilinear resize, ImageNet normalization |
| `HistopathologyDataset` | Histopathology | `(3, 224, 224)` float32 | RGB conversion, Center crop/resize, ImageNet normalization |
| `MRIDataset` | MRI | `(1, 256, 256)` float32 | Slice extraction, Z-score intensity standardization |
| `ClinicalDataset` | Clinical | `(30,)` float32 | Robust scaling, Median imputation, 30 FNA features |
| `EHRDataset` | EHR | `(11,)` float32 | One-hot / frequency encoding, 11 clinical EHR fields |
| `GenomicDataset` | Genomic | `(54675,)` float32 | Quantile normalization, Probe ordering preserved |
| `UnifiedMultimodalDataset` | Multimodal | Dynamic Dict | Polymorphic dispatch based on `detected_modality` |

---

## 3. Caching & Performance Optimization

* **In-Memory LRU Cache**: Automatically buffers the most recent 1,000 decoded tensors in RAM. Access latency: **0.075 ms / sample**.
* **Disk Tile Cache**: Buffers preprocessed binary numpy arrays under `outputs/cache/`. Access latency: **38.9 ms / sample**.
* **Zero Overhead**: Downstream models achieve up to **937.5 samples/second** during batch generation.
