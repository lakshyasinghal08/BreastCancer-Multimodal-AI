# Table 5: Offline Inference Runtime Profiling Across Computational Pipeline Stages

| Pipeline Execution Stage | Operations Description | CPU Latency (ms) | CPU Proportion (%) | GPU Latency (ms) | GPU Proportion (%) |
| --- | --- | --- | --- | --- | --- |
| Modality Payload Ingestion | Parsing DICOM, SVS, and JSON headers | 4.8 | 11.3% | 4.8 | 57.1% |
| Image Resampling & Norm | Spatial interpolation & Z-score scaling | 14.2 | 33.4% | 1.2 | 14.3% |
| Feature Embedding Extraction | Unimodal representations forward pass | 16.5 | 38.8% | 1.6 | 19.0% |
| Dynamic Mask-Aware Fusion | Asymmetric cross-attention & indicator masking | 5.2 | 12.2% | 0.5 | 6.0% |
| Calibration & Risk Stratification | Temperature scaling (T*=1.18) & entropy H(p) | 1.8 | 4.2% | 0.3 | 3.6% |
| Total End-to-End Pipeline | Complete single-patient diagnostic pass | 42.5 | 100.0% | 8.4 | 100.0% |
