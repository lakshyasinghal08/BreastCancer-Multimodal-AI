| Modality                   | Raw Format                  | Tensor Shape   |   Channels | PyTorch Dtype   |   Memory Footprint (KB) | Normalization          |
|:---------------------------|:----------------------------|:---------------|-----------:|:----------------|------------------------:|:-----------------------|
| Mammography                | DICOM / 16-bit PNG          | [1, 512, 512]  |          1 | torch.float32   |                 1024    | Zero-Centered [-1, 1]  |
| Ultrasound                 | 8-bit PNG                   | [1, 256, 256]  |          1 | torch.float32   |                  256    | Zero-Centered [-1, 1]  |
| Histopathology             | RGB PNG Patches             | [3, 224, 224]  |          3 | torch.float32   |                  588    | ImageNet (mean/std)    |
| Magnetic Resonance         | DCE-MRI Series              | [1, 256, 256]  |          1 | torch.float32   |                  256    | Zero-Centered [-1, 1]  |
| Clinical FNA               | Continuous Tabular          | [30]           |         30 | torch.float32   |                    0.12 | Z-Score (RobustScaler) |
| Electronic Health Record   | Mixed Tabular / Categorical | [16]           |         16 | torch.float32   |                    0.06 | Z-Score + One-Hot      |
| Genomics / Transcriptomics | RNA-seq & CNV Probes        | [331]          |        331 | torch.float32   |                    1.32 | Log2 + Standardized    |