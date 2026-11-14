# Multi-Head Attention & Gating Layer Component Breakdown

| Submodule                | Matrix_Dimensions   |   Parameters | Role                                          | Logic                    |
|:-------------------------|:--------------------|-------------:|:----------------------------------------------|:-------------------------|
| Query Projection W_q     | 128 x 128           |        16512 | Transforms token into query representation    | nan                      |
| Key Projection W_k       | 128 x 128           |        16512 | Transforms token into key for dot-product     | nan                      |
| Value Projection W_v     | 128 x 128           |        16512 | Encodes value vectors for attention weighting | nan                      |
| Attention Key Masking    | nan                 |            0 | Zero attention weight for absent modalities   | Add -1e9 where mask == 0 |
| Output Projection W_o    | 128 x 128           |        16512 | Multi-head recombination                      | nan                      |
| Modality Importance Gate | 128 x 32 -> 32 x 1  |         4161 | Computes transparent per-modality weights     | nan                      |
