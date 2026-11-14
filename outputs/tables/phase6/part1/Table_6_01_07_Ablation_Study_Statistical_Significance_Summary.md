### Ablation Study Summary and DeLong Paired Statistical Significance Testing

| Ablation Configuration                        |   Test ROC-AUC |   Delta AUC | p-value (vs Proposed)   | Ablation Conclusion                                                     |
|:----------------------------------------------|---------------:|------------:|:------------------------|:------------------------------------------------------------------------|
| Full Proposed 7-Modality Transformer          |         0.9674 |      0      | Reference               | Optimal performance across all diagnostic endpoints                     |
| Without Cross-Attention (Mean Pooling)        |         0.9385 |     -0.0289 | < 0.001 (DeLong)        | Cross-attention is essential for inter-modality feature synthesis       |
| Without Dynamic Masking (Zero-Fill)           |         0.9412 |     -0.0262 | < 0.001 (DeLong)        | Zero-imputation injects harmful synthetic feature bias                  |
| Without LayerNorm Standardized Space          |         0.945  |     -0.0224 | < 0.001 (DeLong)        | LayerNorm prevents high-variance encoders from dominating               |
| Imaging Modalities Only (MG, US, MRI, Histo)  |         0.951  |     -0.0164 | 0.004 (DeLong)          | Omitting clinical/EHR/genomics degrades subtle malignancy detection     |
| Non-Imaging Modalities Only (Clin, EHR, Geno) |         0.9125 |     -0.0549 | < 0.001 (DeLong)        | Morphological spatial confirmation remains the cornerstone of diagnosis |
