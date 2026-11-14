# Model Card: Dynamic Mask-Aware Cross-Attention Multimodal Transformer

## 1. Model Details
- **Model Name**: Dynamic Mask-Aware Multimodal Breast Cancer Transformer (BCA-MMT-v1.0)
- **Architecture**: Asymmetric cross-attention fusion with dynamic missing modality indicator masking.
- **Developers**: Multimodal Breast Cancer AI Collaborative Research Consortium.
- **Release Date**: November 2026.
- **Model Version**: 1.0.0 (Production Certified).
- **Format**: PyTorch (`.pt`), TorchScript (`.pt`), ONNX Runtime (`.onnx`).
- **License**: Institutional Research & Clinical Evaluation License.
- **Primary Contact**: Research Informatics & Translational AI Unit.

---

## 2. Intended Use & Clinical Scope
- **Primary Clinical Application**: Computer-aided diagnosis (CADx) and risk stratification of breast lesions in adult female patients ($18+$ years).
- **Input Modalities**: 1 to 7 concurrent diagnostic modalities (Mammography, DCE-MRI, Ultrasound, Histopathology WSI, EHR, Genomics, Biomarkers).
- **Primary Output**: Calibrated probability of malignancy $p \in [0, 1]$, predictive uncertainty entropy $H \in [0, 1]$, and modality-specific cross-attention attribution weights.
- **Out-of-Scope Uses**: Sole autonomous decision-making without physician oversight; non-mammary solid tumors; pediatric patient populations.

---

## 3. Quantitative Performance Summary (Test Cohort $N=574$)
- **Area Under ROC Curve (ROC-AUC)**: $0.9674$ ($95\%\text{ CI}: 0.9528 - 0.9806$)
- **Area Under Precision-Recall Curve (PR-AUC)**: $0.9542$ ($95\%\text{ CI}: 0.9365 - 0.9691$)
- **Diagnostic Accuracy**: $91.11\%$ ($523 / 574$ patients)
- **Diagnostic Sensitivity**: $88.79\%$ ($198 / 223$ malignant cases)
- **Diagnostic Specificity**: $92.59\%$ ($325 / 351$ benign cases)
- **Dense Parenchyma Rescue**: ROC-AUC $0.9615$ in BI-RADS c/d ($+0.0950$ gain over solitary mammography)
- **Post-Calibration ECE**: $0.0367$ (Optimal temperature $T^* = 1.18$)
- **Runtime Latency**: $8.4\text{ ms}$ (NVIDIA A100 GPU), $42.5\text{ ms}$ (Intel Xeon CPU).

---

## 4. Ethical, Bias & Subgroup Considerations
- **Ethnic & Demographic Parity**: Validated across African American, Asian, Caucasian, and Hispanic cohorts with demographic parity differences $< 0.025$ and equalized odds disparity $< 0.030$.
- **Density Parity**: Evaluated across all 4 BI-RADS breast density categories (a: fatty, b: scattered, c: heterogeneously dense, d: extremely dense) with minimal variance (AUC range: $0.9592 - 0.9710$).
- **Missing Modality Robustness**: System preserves high diagnostic fidelity even with 5 out of 7 modalities missing ($AUC \ge 0.8950$).
