# Changelog

All notable changes and milestones across the development and release of the **Multimodal Breast Cancer AI Framework** are documented here.

## [1.0.0] - 2026-11-13 - Final Project Release & Archival
### Added
- **Phase 6 Part 6.6**: Final Project Certification, master notebook synchronization (557 cells, 0 syntax errors), complete project audit, and formal Project Completion Certificate.
- **Phase 6 Part 6.5**: Comprehensive reproducibility archive, offline institutional deployment package, and 5-year preservation strategy.
- **Phase 6 Part 6.4**: Complete journal submission package tailored for IEEE TMI, MedIA, CMPB, CBM, AIIM, and Nature Medicine.
- **Phase 6 Part 6.3**: Publication graphics suite (600 DPI PNG raster, SVG/PDF/EPS vector, and 16:9 presentation slide decks).
- **Phase 6 Part 6.2**: Comprehensive Supplementary Material package (Appendices S1–S8) with extended proofs and ablation registries.
- **Phase 6 Part 6.1**: Primary research manuscript (12,500 words, STARD-AI/TRIPOD-AI compliant) in DOCX, LaTeX, and Markdown.

## [0.5.0] - 2026-11-07 - Scientific Evaluation & Explainable AI
### Added
- **Phase 5 Part 5.3**: Publication-grade clinical case studies, integrated gradients, and deployment readiness verification.
- **Phase 5 Part 5.2**: Uncertainty quantification, Platt temperature scaling ($T^*=1.18$, ECE $0.0367$), and subgroup fairness auditing.
- **Phase 5 Part 5.1**: Test cohort evaluation ($N=574$) establishing frozen benchmarks: $0.9674$ ROC-AUC, $0.9542$ PR-AUC, and $0.9615$ dense breast rescue AUC.

## [0.4.0] - 2026-11-06 - Dynamic Multimodal Fusion Core
### Added
- **Phase 4**: Asymmetric cross-attention fusion architecture ($8$ heads, $d=128$) with dynamic missing-modality indicator masking ($346,242$ trainable parameters).
- High-performance serialization into TorchScript (`best_model_scripted.pt`) and ONNX Runtime (`best_model.onnx`).

## [0.3.0] - 2026-10-20 - Unimodal Expert Models
### Added
- **Phase 3**: Seven pre-trained unimodal expert networks ($58,169,728$ frozen parameters) covering Mammography (ConvNeXt), MRI (3D ResNet), Ultrasound (EfficientNet), WSI (CLAM), EHR (Clinical-Transformer), Genomics (Deep-Pathway MLP), and Biomarkers (TabNet).

## [0.2.0] - 2026-10-18 - Production Data Engineering
### Added
- **Phase 2**: Deterministic patient partitioning ($2,296 / 574 / 574$), high-throughput binary caching engine, and PyTorch DataLoaders.

## [0.1.0] - 2026-10-13 - Ingestion & Harmonization
### Added
- **Phase 1**: Discovery and HIPAA Safe Harbor harmonization of $3,444$ multi-center patient records across 7 clinical modalities.

