| Test_Case_Configuration | Active_Modalities | Logits_Output | Fused_Embedding | Attention_Matrix | Prob_Sum | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Mammography Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Ultrasound Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Histopathology Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Breast MRI Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Clinical FNA Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| EHR Staging Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Genomic Only | 1 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Dual: Mammo + Ultrasound | 2 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Dual: Clinical + EHR | 2 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Dual: Histopath + Genomic | 2 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Triple: Mammo + US + Histo | 3 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Triple: MRI + Clinical + Genomic | 3 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Quad: Mammo + MRI + Clinical + EHR | 4 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
| Full Multimodal (All 7 Present) | 7 | [4, 2] | [128] | [4, 4, 8, 8] | 1.0000 | PASSED |
