| Metadata Item              | Specification                                           | Verification Record       |
|:---------------------------|:--------------------------------------------------------|:--------------------------|
| Experiment Identifier      | Phase3_Part4_Production_Training                        | CERTIFIED                 |
| Target Neural Architecture | LightweightMultimodalNet (7-Modality Harmonized)        | 817,189 Parameters        |
| Master Patient Separation  | 5-Fold GroupKFold (Certified 0% Patient Leakage)        | PASS                      |
| Training Dataset Cohort    | 1,889 Unique Human Patients                             | Real Multi-Center Cohort  |
| Validation Dataset Cohort  | 405 Unique Human Patients                               | Real Multi-Center Cohort  |
| Optimization Algorithm     | AdamW (betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-4) | Differential Param Groups |
| LR Scheduling Protocol     | Warmup Cosine Annealing (T_0=2, total=10, min_lr=1e-6)  | Deterministic Schedule    |
| Mixed Precision Engine     | PyTorch AMP Autocast (BFloat16 / FP16 Zero Overhead)    | PASS                      |
| Loss Formulation           | LabelSmoothingCrossEntropy (smoothing=0.05)             | PASS                      |
| Reproducibility Seed       | Seed 42 (PyTorch, NumPy, CUDA)                          | Bitwise Deterministic     |