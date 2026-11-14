| Hyperparameter Dimension    | Configuration Value            | Optimization Rationale                                      |
|:----------------------------|:-------------------------------|:------------------------------------------------------------|
| Global Random Seed          | 42 (Bitwise Deterministic)     | Full reproducible convergence across environments           |
| Base Optimization Algorithm | AdamW (Decoupled Weight Decay) | Robust transformer convergence with L2 regularization       |
| Base Learning Rate          | 5.0e-4 (Peak Warmup Floor)     | Optimal learning rate for small-parameter cross-attention   |
| Weight Decay Factor         | 1.0e-4 (Excluding Norm/Bias)   | Prevents over-parameterization without damping gating       |
| Learning Rate Scheduler     | WarmupCosineAnnealingLR        | Smooth 2-epoch linear warmup + 8-epoch cosine descent       |
| Warmup Epochs               | 2 Epochs (1.0e-4 -> 5.0e-4)    | Stabilizes initial cross-attention query-key projections    |
| Minimum Learning Rate       | 1.0e-6 (Terminal Floor)        | Fine-grained parameter refinement in terminal epochs        |
| Batch Size                  | 32 Multimodal Patient Records  | Balanced gradient variance with CPU/GPU memory footprint    |
| Gradient Accumulation       | 1 Step (Effective Batch = 32)  | Direct synchronous parameter updates                        |
| Gradient Clipping Threshold | 1.000 Maximum L2 Norm          | Eliminates gradient explosion during early attention tuning |
| Label Smoothing Factor      | epsilon = 0.05                 | Prevents overconfident probability calibration spikes       |
| Exponential Moving Average  | beta = 0.999 Shadow Decay      | Smooth parameter trajectory improving generalization        |