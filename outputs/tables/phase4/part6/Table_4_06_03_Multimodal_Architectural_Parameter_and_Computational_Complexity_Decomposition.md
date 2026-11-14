| System Submodule                 | Neural Architecture             |   Parameter Count | Param Share   | Training Status           |   FLOPs (M) | FLOP Share   |
|:---------------------------------|:--------------------------------|------------------:|:--------------|:--------------------------|------------:|:-------------|
| Mammography Vision Encoder       | ResNet-50 (Modified)            |           3654144 | 24.15%        | Frozen (Phase 3)          |       88.5  | 23.83%       |
| Ultrasound Vision Encoder        | DenseNet-121 (Modified)         |           3421056 | 22.61%        | Frozen (Phase 3)          |       84.2  | 22.67%       |
| Histopathology Vision Encoder    | ViT-B/16 Patch Backbone         |           3842560 | 25.39%        | Frozen (Phase 3)          |       96.4  | 25.96%       |
| Breast MRI Vision Encoder        | 3D ResNet-18 Kinetic Extractor  |           3512320 | 23.21%        | Frozen (Phase 3)          |       94.1  | 25.34%       |
| Clinical Tabular MLP             | Residual Tabular MLP            |            118528 | 0.78%         | Frozen (Phase 3)          |        2.8  | 0.75%        |
| EHR Longitudinal MLP             | Temporal Residual MLP           |            124672 | 0.82%         | Frozen (Phase 3)          |        2.9  | 0.78%        |
| Genomic Microarray MLP           | 500-Gene Oncogenic MLP          |            132864 | 0.88%         | Frozen (Phase 3)          |        1.31 | 0.35%        |
| Cross-Attention Transformer Core | 2-Layer 4-Head Transformer      |            263424 | 1.74%         | Trainable (Phase 4)       |        0.94 | 0.25%        |
| Global Diagnostic [CLS] Token    | Learnable 128-d Vector          |               128 | 0.00%         | Trainable (Phase 4)       |        0    | 0.00%        |
| Modality Positional Embeddings   | 7 x 128-d Type Embeddings       |               896 | 0.01%         | Trainable (Phase 4)       |        0    | 0.00%        |
| Malignancy Binary Head           | LayerNorm + Linear (128->2)     |               258 | 0.00%         | Trainable (Phase 4)       |        0.08 | 0.02%        |
| Molecular PAM50 Subtype Head     | LayerNorm + Linear (128->5)     |               645 | 0.00%         | Trainable (Phase 4)       |        0.16 | 0.04%        |
| TOTAL MULTIMODAL SYSTEM          | Unified Cross-Attention Network |          15132806 | 100.00%       | 326,662 Trainable (2.16%) |      371.39 | 100.00%      |