# Master Pipeline Execution Order & Flow

The master notebook executes sequentially across 44 sections:

1. **Sections 1–4**: Environment verification, hardware profiling, deterministic seeding, logging initialization.
2. **Sections 5–8**: Dataset discovery, DICOM/WSI ingestion, patient-level verification, manifest compilation.
3. **Sections 9–13**: Patient-level train/val/test splitting ($N=2,296 / 574 / 574$), data augmentation pipelines.
4. **Sections 14–17**: Feature engineering, high-throughput caching engine, PyTorch DataLoaders.
5. **Sections 18–24**: Seven unimodal expert model definitions (ConvNeXt, 3D ResNet, EfficientNet, Histo-ResNet, Clinical-Transformer, Deep-Pathway MLP, TabNet).
6. **Sections 25–27**: Unimodal training, evaluation, weight freezing, and checkpoint serialization.
7. **Sections 28–31**: Dynamic mask-aware cross-attention fusion architecture, multi-task heads.
8. **Sections 32–34**: End-to-end fusion training, loss optimization, checkpointing.
9. **Sections 35–37**: Test cohort evaluation ($N=574$), bootstrap confidence intervals, ROC-AUC ($0.9674$).
10. **Sections 38–39**: Explainable AI (Grad-CAM, Integrated Gradients, cross-attention attribution), calibration ($T^*=1.18$).
11. **Sections 40–43**: Phase 6 Parts 6.1–6.4 generation (Manuscript, Supplementary, Graphics, Submission).
12. **Section 44 (Subsections 44.1–44.12)**: Complete Reproducibility Archive, Deployment Package, Long-Term Preservation & Final Project Master Certification.
