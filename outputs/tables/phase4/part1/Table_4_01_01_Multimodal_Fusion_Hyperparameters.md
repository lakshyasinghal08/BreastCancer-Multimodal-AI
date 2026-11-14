| Hyperparameter | Configured_Value | Search_Range / Alternatives | Clinical & Architectural Rationale |
| --- | --- | --- | --- |
| Latent Manifold Dimension (d_embed) | 128 | [64, 128, 256] | Standardized latent dimension matching all 7 Phase 3 unimodal encoders |
| Number of Attention Heads (H) | 4 | [2, 4, 8] | Enables 4 distinct cross-modal attention projections with d_k = 32 |
| Head Dimension (d_k) | 32 | [16, 32, 64] | Optimized ratio preserving numerical stability and gradient flow |
| Transformer Layers (L) | 2 | [1, 2, 4] | Compact 2-stage cross-attention refinement preventing over-smoothing |
| Feed-Forward Dimension (d_ff) | 256 | [128, 256, 512] | Provides 2x bottleneck expansion for non-linear feature interaction |
| Dropout Regularization | 0.15 | [0.10, 0.20] | Prevents co-adaptation across multi-modal token pathways |
| Sequence Length (S) | 8 | 7 Modalities + 1 CLS | Standardized token sequence with global classification anchor token |
| Confidence Weighting Gating | Tanh Gating | Linear, Sigmoid, Tanh | Smooth adaptive modulation of modality token magnitude |
