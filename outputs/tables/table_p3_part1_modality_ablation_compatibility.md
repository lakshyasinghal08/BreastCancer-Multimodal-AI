# Cross-Modal Ablation Study Compatibility Verification

| Ablation_Setting           |   Active_Modalities | Output_Shape   |   Latency_ms | Compatibility    |
|:---------------------------|--------------------:|:---------------|-------------:|:-----------------|
| Complete Model (All 7)     |                   7 | (B, 2)         |         38.2 | 100% OPERATIONAL |
| Dual Imaging (Mammo + US)  |                   2 | (B, 2)         |         38.1 | 100% OPERATIONAL |
| Vision + Genomics          |                   3 | (B, 2)         |         38.2 | 100% OPERATIONAL |
| Clinical + EHR Only        |                   2 | (B, 2)         |         38   | 100% OPERATIONAL |
| Single Modality Diagnostic |                   1 | (B, 2)         |         37.9 | 100% OPERATIONAL |
