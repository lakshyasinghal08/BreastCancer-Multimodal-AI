# Neural Network Parameter Allocation & Computational Budget

| Subsystem                       |   Parameters | Percentage   | FLOPs_Estimate   | Status                |
|:--------------------------------|-------------:|:-------------|:-----------------|:----------------------|
| Vision Encoders (4 Modalities)  |       579680 | 70.9%        | 0.78 GFLOPs      | OPTIMIZED             |
| Genomic Microarray Encoder      |        59520 | 7.3%         | 0.06 GFLOPs      | OPTIMIZED             |
| Clinical & EHR Tabular Encoders |        26624 | 3.3%         | 0.02 GFLOPs      | OPTIMIZED             |
| Embedding Harmonizer & Tokens   |         2048 | 0.3%         | < 0.01 GFLOPs    | OPTIMIZED             |
| Cross-Modal Attention Fusion    |       136641 | 16.7%        | 0.14 GFLOPs      | OPTIMIZED             |
| Diagnostic Classifier Head      |        12676 | 1.6%         | < 0.01 GFLOPs    | OPTIMIZED             |
| Total Model Architecture        |       817189 | 100.0%       | 1.01 GFLOPs      | LIGHTWEIGHT CERTIFIED |
