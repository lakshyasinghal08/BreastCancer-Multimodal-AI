| Subsystem Component                    | Execution Device   |   Latency (ms) |   Share (%) | Overhead Type                |
|:---------------------------------------|:-------------------|---------------:|------------:|:-----------------------------|
| PACS DICOM Ingestion & Header Parse    | CPU I/O            |           8.4  |       10.92 | Network & Storage            |
| Preprocessing (Resize, Norm, Align)    | CPU                |           6.2  |        8.06 | Data Ingestion               |
| Mammography Vision Expert Forward      | CPU / GPU          |          10.15 |       13.2  | Vision Feature Extraction    |
| Ultrasound Vision Expert Forward       | CPU / GPU          |           8.45 |       10.99 | Vision Feature Extraction    |
| Histopathology Vision Expert Forward   | CPU / GPU          |          10.8  |       14.04 | Vision Feature Extraction    |
| Breast MRI 3D Vision Expert Forward    | CPU / GPU          |          12.5  |       16.25 | Vision Feature Extraction    |
| Clinical FNA Cytology MLP Forward      | CPU                |           0.45 |        0.59 | Tabular Feature Extraction   |
| EHR Tabular Transformer Forward        | CPU                |           0.65 |        0.85 | Tabular Feature Extraction   |
| Genomic Microarray MLP Forward         | CPU                |           1.8  |        2.34 | Genomic Feature Extraction   |
| Modality Token Embedding & Gating      | CPU / GPU          |           0.85 |        1.11 | Multimodal Fusion            |
| Cross-Attention Transformer (2 Layers) | CPU / GPU          |           3.2  |        4.16 | Multimodal Fusion            |
| Dual-Head Classifier & Subtyping       | CPU / GPU          |           0.95 |        1.24 | Classification & Uncertainty |
| Real-Time XAI (Rollout & Grad-CAM)     | CPU / GPU          |          12.5  |       16.25 | Clinical Explainability      |
| TOTAL END-TO-END DIAGNOSTIC PIPELINE   | Unified System     |          76.9  |      100    | Full Patient Workup          |