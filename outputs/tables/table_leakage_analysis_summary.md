# Summary of Patient Leakage Audit and Protective Mitigation Policies

| Threat_Vector                       | Observed_Frequency       | Risk_Level                                  | Mitigation_Policy                              |
|:------------------------------------|:-------------------------|:--------------------------------------------|:-----------------------------------------------|
| Intra-Dataset Multi-Tile Inflation  | 229 IDC pts, 122 VGF pts | Severe (Cross-split tile contamination)     | Enforce Patient-Level GroupKFold CV            |
| Cross-Dataset Patient Linkage       | 47 TCGA patients         | Benign (Authentic multimodal co-occurrence) | Integrate in joint multimodal evaluation       |
| Institutional Identifier Collision  | 1 patient (8913)         | Moderate (False linkage of distinct cases)  | Enforce Cohort Prefix (CLIN_8913 vs HIST_8913) |
| Duplicate Image References          | 0 duplicates             | None                                        | Verified SHA-256 fingerprinting                |
| Repeated Genomic / Clinical Samples | 0 unaligned duplications | None                                        | Manifest sample indexing                       |
