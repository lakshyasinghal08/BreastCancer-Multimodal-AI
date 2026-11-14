# Modality-Specific Feature Encoder Specifications

| Modality            | Input_Dimensions   | Architecture_Type                      |   Latent_Output |   Parameters | Sparsity_Support   |
|:--------------------|:-------------------|:---------------------------------------|----------------:|-------------:|:-------------------|
| Mammography         | 1 x 224 x 224      | Depthwise-Separable CNN                |             128 |      144,848 | Mask-Aware         |
| Ultrasound          | 1 x 224 x 224      | Depthwise-Separable CNN                |             128 |      144,848 | Mask-Aware         |
| Histopathology      | 3 x 224 x 224      | Depthwise-Separable CNN                |             128 |      145,136 | Mask-Aware         |
| MRI                 | 1 x 224 x 224      | Depthwise-Separable CNN                |             128 |      144,848 | Mask-Aware         |
| Clinical FNA        | 30                 | Residual MLP (30->64->128)             |             128 |       14,656 | Mask-Aware         |
| TCGA EHR            | 16                 | Residual Tabular (16->64->128)         |             128 |       11,968 | Mask-Aware         |
| Genomics (GSE45827) | 331                | Bottleneck Compression (331->128->128) |             128 |       59,520 | Mask-Aware         |
