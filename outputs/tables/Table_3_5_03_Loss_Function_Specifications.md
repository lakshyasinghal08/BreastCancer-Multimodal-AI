| Loss Formulation                        | Hyperparameters             | Class Weighting      | Target Domain               | Regularization Property            |
|:----------------------------------------|:----------------------------|:---------------------|:----------------------------|:-----------------------------------|
| Label Smoothing Cross-Entropy (Default) | epsilon = 0.05              | Supported            | Multimodal Diagnosis        | Prevents Logit Saturation          |
| Weighted Binary Cross-Entropy           | pos_weight auto-fitted      | Auto (n_neg / n_pos) | Class Imbalance Correction  | Penalizes Minority False Negatives |
| Focal Loss                              | alpha=0.25, gamma=2.0       | Modulated            | Hard Example Mining         | Downweights Easy Examples          |
| Dice Loss Placeholder                   | smooth = 1.0                | Spatial              | Segmentation & Localization | Overlap Maximization               |
| Composite Hybrid Loss                   | 0.7*LabelSmooth + 0.3*Focal | Composite            | Joint Diagnostic Robustness | Multi-Objective Optimization       |