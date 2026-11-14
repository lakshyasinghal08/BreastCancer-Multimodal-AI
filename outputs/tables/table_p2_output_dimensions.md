| Modality                 | Tensor_Shape            | Dtype   |   Memory_Per_Sample_KB | Batch_Shape_32              |
|:-------------------------|:------------------------|:--------|-----------------------:|:----------------------------|
| CBIS-DDSM Mammography    | [1, 512, 512]           | float32 |                1024    | [32, 1, 512, 512]           |
| BUSI Ultrasound          | [1, 256, 256]           | float32 |                 256    | [32, 1, 256, 256]           |
| BreakHis Histopathology  | [3, 224, 224]           | float32 |                 588    | [32, 3, 224, 224]           |
| Breast DCE-MRI           | [1, 256, 256]           | float32 |                 256    | [32, 1, 256, 256]           |
| Wisconsin Clinical FNA   | [30]                    | float32 |                   0.12 | [32, 30]                    |
| TCGA-BRCA EHR            | [18]                    | float32 |                   0.07 | [32, 18]                    |
| Genomic Profile          | [1000]                  | float32 |                   4    | [32, 1000]                  |
| Multimodal Patient Tuple | Aligned Tuple + Mask[7] | float32 |                2128.2  | Collatable Multimodal Batch |