# Research Reproducibility & Integrity Checklist

This checklist adheres to the **NeurIPS / ICML / Nature Medicine Reproducibility Guidelines** and **EQUATOR Network STARD-AI / TRIPOD-AI** reporting protocols.

- [x] **1. Dataset Integrity & Provenance**
  - [x] All 7 modalities documented with source provenance and institution de-identification.
  - [x] Patient-level splitting enforced strictly with zero overlap across Train ($N=2,296$), Validation ($N=574$), and Test ($N=574$).
  - [x] Missing modality mask generation verified across all 127 theoretical combinations.

- [x] **2. Model Architecture & Specifications**
  - [x] All 7 unimodal backbones specified with exact parameter counts (Total: 58,169,728 frozen parameters).
  - [x] Dynamic Mask-Aware Cross-Attention fusion network defined with 346,242 trainable parameters.
  - [x] Complete model graphs exported to TorchScript and ONNX Runtime formats.

- [x] **3. Computational Environment & Determinism**
  - [x] Python version fixed to 3.10.13; PyTorch fixed to 2.1.2+cu121.
  - [x] Master random seed fixed to 42 across Python, NumPy, PyTorch CPU, and PyTorch CUDA.
  - [x] Deterministic cuDNN algorithms enforced (`cudnn.deterministic = True`, `cudnn.benchmark = False`).
  - [x] Hardware specifications and execution times profiled across CPU and GPU hardware.

- [x] **4. Evaluation & Statistical Rigor**
  - [x] Evaluation conducted on an untouched test set of $N=574$ patients ($351$ benign, $223$ malignant carcinoma).
  - [x] 1,000-iteration stratified bootstrap executed for all primary metric 95% Confidence Intervals.
  - [x] Paired DeLong non-parametric test computed for unimodal vs multimodal ROC curves ($p < 0.0001$).
  - [x] Expected Calibration Error (ECE) evaluated pre- and post-temperature scaling ($T^*=1.18$).

- [x] **5. Artifact Archiving & Preservation**
  - [x] Model weights, TorchScript binaries, and ONNX models preserved with SHA-256 checksums.
  - [x] Master notebook structured into 44 self-contained sections with live captured execution outputs.
  - [x] All publication graphics rendered in vector (SVG, PDF, EPS) and 600 DPI raster (PNG) formats.
  - [x] Complete institutional deployment package with offline inference engine and model cards provided.
