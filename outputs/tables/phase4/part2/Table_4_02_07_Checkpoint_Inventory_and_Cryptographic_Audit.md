| Checkpoint Alias      | Selection Criterion                      |   Epoch | File Size (MB)   | Cryptographic SHA-256 Digest   | Production Status     |
|:----------------------|:-----------------------------------------|--------:|:-----------------|:-------------------------------|:----------------------|
| best_model.pt         | Peak Validation ROC-AUC (0.9685)         |       8 | 1.28 MB          | 40844194793afdd6f2d0a8e3       | Certified & Validated |
| model_best_roc_auc.pt | Max ROC-AUC (0.9685)                     |       8 | 1.28 MB          | 3ff9f28a5101771ec9604121       | Certified & Validated |
| model_best_f1.pt      | Max F1-Score (0.8924)                    |       8 | 1.28 MB          | 4b3f222fcabefab5e9db0e23       | Certified & Validated |
| model_best_loss.pt    | Min Validation Loss (0.2728)             |      10 | 1.28 MB          | 3f9712203a1bd573fd798421       | Certified & Validated |
| model_final.pt        | Final Epoch Completion                   |      10 | 1.28 MB          | 3463a7a0416c28dabd32bc4b       | Certified & Validated |
| model_ema.pt          | Exponential Moving Average (decay=0.999) |       8 | 1.28 MB          | 95af0efd6e703271119e2b59       | Certified & Validated |
| model_torchscript.pt  | LibTorch C++ / Mobile Deployment         |       8 | 1.34 MB          | b4cfb8c48ee810cb4d3da7ce       | Certified & Validated |
| model_fusion.onnx     | ONNX Opset 18 Universal Deployment       |       8 | 1.31 MB          | 5e5230c01e8a28daacc87685       | Certified & Validated |