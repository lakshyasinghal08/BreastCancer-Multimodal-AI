| Source Modality   | Target Modality   |   Layer 1 Attention |   Layer 2 Attention |   Attention Rollout Flow | Clinical Synergy                  |
|:------------------|:------------------|--------------------:|--------------------:|-------------------------:|:----------------------------------|
| Ultrasound        | Histopathology    |                0.22 |                0.25 |                   0.1722 | High Imaging-Pathology Cross-Talk |
| Mammography       | Histopathology    |                0.2  |                0.24 |                   0.1641 | High Imaging-Pathology Cross-Talk |
| Ultrasound        | Mammography       |                0.21 |                0.22 |                   0.1592 | High Imaging-Pathology Cross-Talk |
| Breast MRI        | Histopathology    |                0.19 |                0.22 |                   0.1552 | High Imaging-Pathology Cross-Talk |
| Histopathology    | Mammography       |                0.18 |                0.19 |                   0.142  | High Imaging-Pathology Cross-Talk |
| Mammography       | Ultrasound        |                0.18 |                0.2  |                   0.1394 | High Imaging-Pathology Cross-Talk |
| Genomics          | Histopathology    |                0.15 |                0.18 |                   0.1294 | Molecular-Morphology Cross-Talk   |
| Histopathology    | Ultrasound        |                0.16 |                0.17 |                   0.1252 | High Imaging-Pathology Cross-Talk |
| Breast MRI        | Mammography       |                0.15 |                0.16 |                   0.1239 | High Imaging-Pathology Cross-Talk |
| Clinical          | Histopathology    |                0.14 |                0.16 |                   0.1209 | Clinical Baseline Context         |
| EHR               | Clinical          |                0.14 |                0.12 |                   0.0982 | Clinical Baseline Context         |
| EHR               | Genomics          |                0.12 |                0.14 |                   0.098  | Molecular-Morphology Cross-Talk   |
| Breast MRI        | Ultrasound        |                0.11 |                0.12 |                   0.0957 | High Imaging-Pathology Cross-Talk |
| Histopathology    | Breast MRI        |                0.12 |                0.11 |                   0.0912 | High Imaging-Pathology Cross-Talk |
| Mammography       | Breast MRI        |                0.11 |                0.12 |                   0.0908 | High Imaging-Pathology Cross-Talk |
| Clinical          | Mammography       |                0.1  |                0.11 |                   0.0907 | Clinical Baseline Context         |
| Genomics          | Clinical          |                0.12 |                0.11 |                   0.0848 | Molecular-Morphology Cross-Talk   |
| Ultrasound        | Breast MRI        |                0.09 |                0.1  |                   0.0793 | High Imaging-Pathology Cross-Talk |
| Genomics          | Mammography       |                0.08 |                0.09 |                   0.0787 | Molecular-Morphology Cross-Talk   |
| EHR               | Histopathology    |                0.08 |                0.09 |                   0.0785 | Clinical Baseline Context         |
| Genomics          | Breast MRI        |                0.09 |                0.1  |                   0.0782 | Molecular-Morphology Cross-Talk   |
| Clinical          | Genomics          |                0.1  |                0.1  |                   0.077  | Molecular-Morphology Cross-Talk   |
| Clinical          | Ultrasound        |                0.08 |                0.09 |                   0.0743 | Clinical Baseline Context         |
| Clinical          | Breast MRI        |                0.08 |                0.09 |                   0.0718 | Clinical Baseline Context         |
| Genomics          | Ultrasound        |                0.06 |                0.07 |                   0.0624 | Molecular-Morphology Cross-Talk   |
| EHR               | Mammography       |                0.06 |                0.07 |                   0.0614 | Clinical Baseline Context         |
| Breast MRI        | Genomics          |                0.07 |                0.06 |                   0.0538 | Molecular-Morphology Cross-Talk   |
| Histopathology    | Genomics          |                0.07 |                0.06 |                   0.0533 | Molecular-Morphology Cross-Talk   |
| EHR               | Ultrasound        |                0.05 |                0.06 |                   0.0517 | Clinical Baseline Context         |
| EHR               | Breast MRI        |                0.05 |                0.06 |                   0.0507 | Clinical Baseline Context         |
| Clinical          | EHR               |                0.07 |                0.06 |                   0.0493 | Clinical Baseline Context         |
| Genomics          | EHR               |                0.06 |                0.05 |                   0.0428 | Molecular-Morphology Cross-Talk   |
| Mammography       | Genomics          |                0.05 |                0.04 |                   0.0413 | Molecular-Morphology Cross-Talk   |
| Ultrasound        | Genomics          |                0.05 |                0.04 |                   0.0411 | Molecular-Morphology Cross-Talk   |
| Breast MRI        | Clinical          |                0.05 |                0.04 |                   0.039  | Clinical Baseline Context         |
| Histopathology    | Clinical          |                0.05 |                0.04 |                   0.0386 | Clinical Baseline Context         |
| Mammography       | Clinical          |                0.04 |                0.04 |                   0.0356 | Clinical Baseline Context         |
| Ultrasound        | Clinical          |                0.04 |                0.04 |                   0.0355 | Clinical Baseline Context         |
| Breast MRI        | EHR               |                0.02 |                0.02 |                   0.0186 | Clinical Baseline Context         |
| Mammography       | EHR               |                0.02 |                0.02 |                   0.0183 | Clinical Baseline Context         |
| Ultrasound        | EHR               |                0.02 |                0.02 |                   0.0183 | Clinical Baseline Context         |
| Histopathology    | EHR               |                0.02 |                0.01 |                   0.0152 | Clinical Baseline Context         |