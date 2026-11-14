| Subsystem                            |   Param Count | Grad Norm Range   |   Clipping Threshold | Status                |
|:-------------------------------------|--------------:|:------------------|---------------------:|:----------------------|
| Mammography Encoder (1-ch)           |        181824 | [0.012, 0.450]    |                    1 | HEALTHY (Propagating) |
| Ultrasound Encoder (1-ch)            |        181824 | [0.010, 0.420]    |                    1 | HEALTHY (Propagating) |
| Histopathology Encoder (3-ch)        |        182112 | [0.015, 0.480]    |                    1 | HEALTHY (Propagating) |
| MRI Encoder (1-ch)                   |        181824 | [0.011, 0.410]    |                    1 | HEALTHY (Propagating) |
| Clinical FNA Encoder (30-dim)        |         14592 | [0.005, 0.280]    |                    1 | HEALTHY (Propagating) |
| Longitudinal EHR Encoder (16-dim)    |         12800 | [0.004, 0.250]    |                    1 | HEALTHY (Propagating) |
| Genomic Microarray Encoder (331-dim) |         59392 | [0.008, 0.350]    |                    1 | HEALTHY (Propagating) |
| Harmonizer & Presence Tokens         |          1792 | [0.002, 0.180]    |                    1 | HEALTHY (Propagating) |
| Cross-Modal Multi-Head Self-Attn     |        133248 | [0.020, 0.620]    |                    1 | HEALTHY (Propagating) |
| Multimodal Classifier Head           |         14755 | [0.030, 0.750]    |                    1 | HEALTHY (Propagating) |