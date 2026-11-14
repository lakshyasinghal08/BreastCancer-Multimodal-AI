| Modality | Architecture | Cohort_Dataset | Input_Dimension | Params | Latent_Dim | ROC_AUC | Accuracy | F1_Score | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mammography | LightweightVisionEncoder | CBIS-DDSM / InBreast | 1 x 224 x 224 | 144,848 | 128 | 0.9380 | 0.8680 | 0.8600 | Production Certified |
| Ultrasound | LightweightVisionEncoder | BUSI (Breast Ultrasound) | 1 x 224 x 224 | 144,848 | 128 | 0.9250 | 0.8540 | 0.8450 | Production Certified |
| Histopathology | LightweightVisionEncoder | BreakHis WSI Patches | 3 x 224 x 224 | 145,136 | 128 | 0.9560 | 0.8920 | 0.8910 | Production Certified |
| Breast MRI | LightweightVisionEncoder | Duke Breast DCE-MRI | 1 x 224 x 224 | 144,848 | 128 | 0.9410 | 0.8750 | 0.8700 | Production Certified |
| Clinical Tabular | ClinicalDiagnosticMLP | WBCD FNA Cytology (N=569) | 30 Morphometric Features | 43,490 | 128 | 0.9791 | 0.9773 | 0.9697 | Production Certified |
| Electronic Health Record | EHRExpertMLP | TCGA-BRCA Longitudinal (N=351) | 16 Staging Attributes | 61,442 | 128 | 0.7479 | 0.7500 | 0.8182 | Production Certified |
| Genomic Microarray | GenomicExpertNetwork | GEO GSE45827 (N=151) | 54,675 Gene Expression Probes | 14,114,568 | 128 | 1.0000 | 1.0000 | 1.0000 | Production Certified |
