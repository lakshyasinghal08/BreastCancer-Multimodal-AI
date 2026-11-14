| Pipeline_Stage | Tensor_Name | Tensor_Shape | Data_Type | NaN_Detected | Inf_Detected | Validation_Result |
| --- | --- | --- | --- | --- | --- | --- |
| Unimodal Expert Extraction | raw_tokens | [B, 7, 128] | torch.float32 | False | False | PASSED (Shape Aligned) |
| Confidence Weighting | confidence_weights | [B, 7] | torch.float32 | False | False | PASSED (Sum=1.000) |
| Modality Gating | gated_tokens | [B, 7, 128] | torch.float32 | False | False | PASSED (Tanh Modulated) |
| Token Sequence Injection | seq_tokens | [B, 8, 128] | torch.float32 | False | False | PASSED (CLS Preprended) |
| Sequence Mask Construction | seq_mask | [B, 8] | torch.bool | False | False | PASSED (CLS=True) |
| Cross-Attention Layer 1 | attn_weights_l1 | [B, 4, 8, 8] | torch.float32 | False | False | PASSED (Softmax Normal) |
| Cross-Attention Layer 2 | attn_weights_l2 | [B, 4, 8, 8] | torch.float32 | False | False | PASSED (Softmax Normal) |
| Dual-Pooling Layer | fused_embedding | [B, 128] | torch.float32 | False | False | PASSED (LayerNorm Standard) |
| Diagnostic Head | logits | [B, 2] | torch.float32 | False | False | PASSED (Binary Logits) |
| Probability Calibration | probabilities | [B, 2] | torch.float32 | False | False | PASSED (Sum=1.000) |
| Auxiliary Subtype Head | subtype_logits | [B, 6] | torch.float32 | False | False | PASSED (PAM50 Logits) |
| Diagnostic Confidence | confidence_scores | [B] | torch.float32 | False | False | PASSED (Range [0.5, 1.0]) |
| Predictive Uncertainty | predictive_entropy | [B] | torch.float32 | False | False | PASSED (Entropy >= 0.0) |
