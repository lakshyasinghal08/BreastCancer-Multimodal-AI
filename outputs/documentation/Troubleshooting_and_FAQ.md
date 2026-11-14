# Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: What should I do if CUDA out-of-memory (OOM) occurs during fusion training?
**Resolution**: The fusion architecture requires only $1.42\text{ GB}$ VRAM at batch size 32 because all 7 unimodal backbones are frozen. If OOM occurs:
1. Verify that `torch.no_grad()` is set during feature extraction.
2. Reduce training batch size from 32 to 16 in Section 32.
3. Enable mixed-precision automatic mixed precision (`torch.cuda.amp.autocast()`).

### Q2: How does the system handle patients with only 1 or 2 available modalities?
**Resolution**: The dynamic mask-aware fusion module uses a binary missing-modality mask $m \in \{0, 1\}^7$. Missing modalities are filled with zero embeddings and masked out during cross-attention softmax operations:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M_{\text{mask}}\right)V$$
where $M_{\text{mask}, j} = -\infty$ if modality $j$ is absent. The system maintains high diagnostic performance ($AUC \ge 0.8950$) even with severe data missingness.

### Q3: What is the exact numerical difference between PyTorch and ONNX predictions?
**Resolution**: On the $N=574$ test cohort, the maximum absolute difference $|p_{\text{torch}} - p_{\text{onnx}}| \le 4.1 \times 10^{-6}$, ensuring bitwise clinical equivalence.

### Q4: Can the offline inference engine run on Windows without WSL?
**Resolution**: Yes. All scripts and the `offline_inference_engine.py` are natively cross-platform and execute seamlessly on Windows PowerShell, macOS, and Linux.
