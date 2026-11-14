| Sequence_Index | Token_Type | Source_Modality | Input_Dimension | Latent_Dimension | Identity_Embedding | Positional_Embedding |
| --- | --- | --- | --- | --- | --- | --- |
| Index 0 | Global Anchor | Classification [CLS] | None (Learnable Vector) | 128 | Fixed Global | Position 0 |
| Index 1 | Radiological Imaging | Mammography (CBIS-DDSM) | 1 x 224 x 224 | 128 | Modality ID 1 | Position 1 |
| Index 2 | Sonographic Imaging | Ultrasound (BUSI) | 1 x 224 x 224 | 128 | Modality ID 2 | Position 2 |
| Index 3 | Microscopic Histology | Histopathology (BreakHis) | 3 x 224 x 224 | 128 | Modality ID 3 | Position 3 |
| Index 4 | Volumetric Dynamic | Breast MRI (Duke) | 1 x 224 x 224 | 128 | Modality ID 4 | Position 4 |
| Index 5 | Cytological Tabular | Clinical FNA (WBCD) | 30 Attributes | 128 | Modality ID 5 | Position 5 |
| Index 6 | Longitudinal EHR | Clinical Staging (TCGA) | 16 Attributes | 128 | Modality ID 6 | Position 6 |
| Index 7 | Transcriptomic Genomic | Gene Microarrays (GSE45827) | 54,675 Probes | 128 | Modality ID 7 | Position 7 |
