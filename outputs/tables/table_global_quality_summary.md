# Global Medical Dataset Quality Index Dimension Decomposition

| Dimension                      |   Weight |   Raw_Score |   Weighted_Contribution | Basis                                                         |
|:-------------------------------|---------:|------------:|------------------------:|:--------------------------------------------------------------|
| Dataset Health Score           |     0.2  |        98.5 |                   19.7  | All 7 cohorts accessible with 0 unhandled read faults         |
| Patient Integrity Score        |     0.15 |       100   |                   15    | 1,918 patients verified with strict namespace isolation       |
| Image Integrity Score          |     0.15 |        99.8 |                   14.97 | Valid bit depth, entropy, dynamic range, and resolution       |
| Metadata Completeness Score    |     0.15 |        96.4 |                   14.46 | Standardized manifest with rich modality annotations          |
| Label Ground Truth Score       |     0.15 |       100   |                   15    | 100% verified clinical biopsy / surgical diagnoses            |
| Completeness & Diversity Score |     0.1  |        94.8 |                    9.48 | Full 7-modality coverage with 47 quad-modal bridge pts        |
| Downstream Readiness Score     |     0.1  |        99   |                    9.9  | Leakage-free partitions (Train 70/Val 15/Test 15) & 5-fold CV |
