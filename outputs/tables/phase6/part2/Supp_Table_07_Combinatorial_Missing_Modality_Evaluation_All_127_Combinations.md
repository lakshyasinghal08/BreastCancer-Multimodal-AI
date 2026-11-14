### Combinatorial Missing Modality Evaluation Across Representative Clinical Ingestion Subsets

| Modality Combination Subset     | Active Encoders                     |   Test ROC-AUC | AUC Retention   | Clinical Scenario                               |
|:--------------------------------|:------------------------------------|---------------:|:----------------|:------------------------------------------------|
| All 7 Modalities (Full)         | MG, US, Histo, MRI, Clin, EHR, Geno |         0.9674 | 100.00%         | Tertiary Comprehensive Cancer Center            |
| Drop Mammography                | US, Histo, MRI, Clin, EHR, Geno     |         0.9512 | 98.33%          | Young patient / Pregnancy screening             |
| Drop Ultrasound                 | MG, Histo, MRI, Clin, EHR, Geno     |         0.954  | 98.61%          | Mammography screening with MRI follow-up        |
| Drop Breast MRI                 | MG, US, Histo, Clin, EHR, Geno      |         0.958  | 99.03%          | Standard hospital lacking dedicated breast coil |
| Drop Histopathology Biopsy      | MG, US, MRI, Clin, EHR, Geno        |         0.9565 | 98.87%          | Pre-biopsy non-invasive clinical evaluation     |
| Drop Genomics                   | MG, US, Histo, MRI, Clin, EHR       |         0.961  | 99.34%          | Community hospital without NGS laboratory       |
| Imaging Only (MG + US)          | Mammography, Ultrasound             |         0.942  | 97.37%          | Standard outpatient screening clinic            |
| Non-Imaging (Clin + EHR + Geno) | Clinical, EHR, Genomics             |         0.9125 | 94.33%          | Remote telemedicine consultation                |
