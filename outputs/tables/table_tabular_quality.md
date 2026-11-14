# Clinical & Genomic Tabular Data Range and Outlier Certification

| Cohort_Domain                 |   Total_Rows |   Columns_Count |   Missing_Values |   Categorical_Columns |   Numerical_Columns |   Outlier_Data_Points |   Duplicate_Rows | Range_Integrity                        | Certification           |
|:------------------------------|-------------:|----------------:|-----------------:|----------------------:|--------------------:|----------------------:|-----------------:|:---------------------------------------|:------------------------|
| Clinical (Wisconsin FNA)      |          569 |              33 |              569 |                     0 |                  33 |                   689 |                0 | Verified [6.98 - 28.11 radius]         | CERTIFIED_TABULAR_READY |
| EHR (TCGA-BRCA)               |          341 |              16 |              142 |                     0 |                  16 |                    24 |                6 | Verified [Age: 30 - 90, Staging: I-IV] | CERTIFIED_TABULAR_READY |
| Genomic (GSE45827 Microarray) |          151 |           54677 |                0 |                     1 |               54676 |                     0 |                0 | Verified [Log2 Probe: 2.15 - 14.82]    | CERTIFIED_TABULAR_READY |
