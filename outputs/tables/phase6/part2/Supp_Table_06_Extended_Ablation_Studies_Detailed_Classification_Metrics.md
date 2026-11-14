### Extended Architectural Ablation Studies: Full Classification Metrics on Locked Test Cohort

| Ablation Configuration              |   ROC-AUC |   PR-AUC |   Accuracy |   Sensitivity |   Specificity |   F1-Score |    ECE |
|:------------------------------------|----------:|---------:|-----------:|--------------:|--------------:|-----------:|-------:|
| Full 7-Modality Proposed            |    0.9674 |   0.9542 |     0.9111 |        0.8879 |        0.9259 |     0.8898 | 0.0367 |
| Without Cross-Attention (Mean)      |    0.9385 |   0.912  |     0.8745 |        0.8385 |        0.8974 |     0.8415 | 0.0612 |
| Without Dynamic Masking (Zero-Fill) |    0.9412 |   0.9185 |     0.8815 |        0.8475 |        0.9031 |     0.852  | 0.0585 |
| Without LayerNorm Projection        |    0.945  |   0.924  |     0.8868 |        0.8565 |        0.906  |     0.859  | 0.0542 |
| Imaging Only (MG, US, MRI, Histo)   |    0.951  |   0.9315 |     0.8937 |        0.8655 |        0.9117 |     0.8675 | 0.0495 |
| Non-Imaging Only (Clin, EHR, Geno)  |    0.9125 |   0.884  |     0.8502 |        0.8072 |        0.8775 |     0.812  | 0.0782 |
