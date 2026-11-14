# Table 6: Central Model Checkpoint Registry, Parameter Allocations, and File Sizes

| Model Component | Modality Domain | Neural Architecture | Frozen Parameters | Trainable Parameters | Checkpoint Size | Parity Diff |
| --- | --- | --- | --- | --- | --- | --- |
| Mammography Expert | Digital Mammography (CC/MLO) | ConvNeXt-Base | 87,566,400 | 0 | 334.8 MB | 0.0 |
| DCE-MRI Expert | 3D Dynamic DCE-MRI | 3D ResNet-50 | 46,159,104 | 0 | 176.4 MB | 0.0 |
| Ultrasound Expert | Diagnostic Ultrasound | EfficientNet-B4 | 17,549,448 | 0 | 67.2 MB | 0.0 |
| Histopathology Expert | Biopsy WSI H&E | CLAM / Histo-ResNet | 21,284,608 | 0 | 81.5 MB | 0.0 |
| EHR Clinical Expert | Longitudinal EHR & Labs | Clinical-Transformer | 12,458,880 | 0 | 47.8 MB | 0.0 |
| Genomics Expert | PAM50 RNA-Seq Panels | Deep-Pathway MLP | 3,211,264 | 0 | 12.4 MB | 0.0 |
| Biomarkers Expert | Tabular IHC Markers | TabNet | 1,540,024 | 0 | 5.9 MB | 0.0 |
| Multimodal Fusion Core | 7-Modality Cross-Attn | Cross-Attention Transformer | 58,169,728 | 346,242 | 1.39 MB | <= 4.1e-6 |
| TorchScript Binary | Serialized Fusion Model | JIT C++ Compiled | 58,169,728 | 346,242 | 1.32 MB | 0.0 |
| ONNX Runtime Engine | Cross-Platform C++ Engine | ONNX Graph Specification | 58,169,728 | 346,242 | 0.21 MB | <= 4.1e-6 |
