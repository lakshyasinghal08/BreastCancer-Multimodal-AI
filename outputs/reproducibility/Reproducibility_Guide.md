# End-to-End Reproducibility Guide: Multimodal Breast Cancer AI Framework

## 1. Executive Summary & Purpose
This guide provides an exhaustive, step-by-step protocol for reproducing every computational artifact, unimodal expert checkpoint, multimodal cross-attention fusion model, statistical evaluation, explainability visualization, and publication table in the Multimodal Breast Cancer AI research lifecycle (Phases 1 through 6).

The framework integrates seven clinical and diagnostic modalities:
1. **Digital Mammography (MG)**: High-resolution craniocaudal (CC) and mediolateral oblique (MLO) projections processed via ConvNeXt-Base.
2. **Dynamic Contrast-Enhanced MRI (DCE-MRI)**: 3D volumetric pharmacokinetic series analyzed via 3D ResNet-50.
3. **Breast Ultrasound (US)**: High-frequency acoustic parenchyma scans processed through EfficientNet-B4.
4. **Histopathology Whole-Slide Imaging (WSI)**: Gigapixel H&E biopsies processed using Histo-ResNet with CLAM clustering.
5. **Electronic Health Records (EHR)**: Longitudinal clinical labs, vitals, and comorbidities embedded via Clinical-Transformer.
6. **Genomics / Transcriptomics (GEN)**: Bulk RNA-seq gene expression signatures (PAM50, Oncotype DX) analyzed with Deep-Pathway MLP.
7. **Clinical Biomarkers / Subtypes (CLI)**: Tabular immunohistochemistry markers (ER, PR, HER2, Ki-67) processed via TabNet.

---

## 2. Hardware & Software Requirements
- **Operating System**: Ubuntu 22.04 LTS or Windows 11 Enterprise (64-bit)
- **CPU**: Intel Xeon Platinum 8380 (16 cores minimum) or AMD EPYC 7763
- **RAM**: Minimum 64 GB ECC DDR4/DDR5 (128 GB recommended)
- **GPU**: NVIDIA A100-SXM4-80GB (minimum 1x NVIDIA RTX 3090 / 4090 24GB for inference)
- **CUDA / cuDNN**: CUDA 12.2 / cuDNN 8.9.2
- **Python**: Version 3.10.13

---

## 3. Environment Setup & Dependency Installation
To establish an isolated, bitwise-deterministic virtual environment, execute:

```bash
# 1. Create and activate dedicated conda environment
conda create -n bca_multimodal python=3.10.13 -y
conda activate bca_multimodal

# 2. Install PyTorch ecosystem with matching CUDA 12.1 runtime
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install core scientific and machine learning dependencies
pip install numpy==1.26.2 scipy==1.11.4 pandas==2.1.4 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0

# 4. Install clinical & imaging libraries
pip install pydicom==2.4.4 SimpleITK==2.3.1 opencv-python==4.8.1.78 pillow==10.1.0 onnxruntime==1.16.3

# 5. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA Available:', torch.cuda.is_available())"
```

---

## 4. Deterministic Random Seeds & Parity Configuration
All random number generators must be seeded identically prior to data loading, model weight initialization, or cross-validation partitioning:

```python
import os
import random
import numpy as np
import torch

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 5. End-to-End Pipeline Execution Order
Execution follows a strict linear pipeline across the master notebook `BreastCancer_Multimodal_AI_Master.ipynb`:

| Phase | Master Section | Primary Action | Generated Artifacts |
|---|---|---|---|
| **Phase 1** | Sections 1–8 | Dataset Harmonization & Cohort Verification | `dataset_manifest.json`, `cohort_summary.csv` |
| **Phase 2** | Sections 9–17 | Patient-Level Data Engineering & Caching | Processed modalities, PyTorch DataLoaders |
| **Phase 3** | Sections 18–27 | Unimodal Expert Training & Certification | 7 expert checkpoints (`outputs/models/*_expert.pt`) |
| **Phase 4** | Sections 28–34 | Dynamic Mask-Aware Cross-Attention Fusion | Fusion checkpoint (`outputs/models/best_model.pt`) |
| **Phase 5** | Sections 35–39 | Scientific Evaluation & Statistical Testing | ROC-AUC ($0.9674$), Delong test, calibration |
| **Phase 6** | Sections 40–44 | Manuscript, Graphics, Submission & Archive | Full paper, supplementary, 600 DPI figures, archive |

---

## 6. Model Verification & Numerical Parity Targets
An independent reproduction must replicate the frozen test cohort metrics ($N=574$) within the defined numerical tolerances:

- **Multimodal ROC-AUC**: $0.9674 \pm 0.0005$ ($95\%\text{ CI}: 0.9528 - 0.9806$)
- **PR-AUC**: $0.9542 \pm 0.0008$
- **Diagnostic Accuracy**: $91.11\% \pm 0.15\%$
- **Diagnostic Sensitivity**: $88.79\% \pm 0.20\%$
- **Diagnostic Specificity**: $92.59\% \pm 0.15\%$
- **Expected Calibration Error (ECE)**: $0.0367 \pm 0.0010$ (Temperature scaled at $T^* = 1.18$)
- **Dense Parenchymal Rescue**: $0.9615$ ROC-AUC in BI-RADS c/d ($+0.0950$ gain over solitary mammography)
- **Serialization Parity**: Maximum absolute difference between PyTorch and ONNX predictions $\le 4.1 \times 10^{-6}$.

---

## 7. Artifact Integrity Verification
Run the verification script to audit SHA-256 checksums across all model checkpoints:

```bash
python -c "
import hashlib, json
with open('outputs/artifact_inventory/checksum_manifest.json') as f:
    manifest = json.load(f)
for path, expected_hash in manifest['models'].items():
    with open(path, 'rb') as fp:
        actual_hash = hashlib.sha256(fp.read()).hexdigest()
    assert actual_hash == expected_hash, f'Mismatch in {path}'
print('All model hashes verified successfully!')
"
```
