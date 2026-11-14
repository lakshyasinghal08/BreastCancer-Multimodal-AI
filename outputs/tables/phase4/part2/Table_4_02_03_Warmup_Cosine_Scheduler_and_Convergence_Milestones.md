| Epoch Window   |   Scheduled LR | Phase Function            | Target Behavior                                            |   Observed Val Loss |   Observed Val ROC-AUC |
|:---------------|---------------:|:--------------------------|:-----------------------------------------------------------|--------------------:|-----------------------:|
| Epoch 1        |       0.0001   | Linear Warmup Phase 1     | Query-Key subspace initialization; gentle gradient updates |              0.6512 |                 0.8415 |
| Epoch 2        |       0.0005   | Peak Warmup Acquisition   | Full-capacity exploration; rapid cross-modal alignment     |              0.4431 |                 0.9142 |
| Epoch 3        |       0.000481 | Early Cosine Descent      | Dominant cross-attention routing established               |              0.3621 |                 0.9418 |
| Epoch 4        |       0.000427 | Cosine Deceleration       | Modality gating settles; confidence scores calibrate       |              0.3204 |                 0.9524 |
| Epoch 5        |       0.000346 | Mid-Trajectory Descent    | Fine-tuning cross-modal attention matrices                 |              0.3018 |                 0.9588 |
| Epoch 6        |       0.000251 | Sub-Half Annealing        | Suppression of spurious unimodal correlations              |              0.2894 |                 0.9625 |
| Epoch 7        |       0.000155 | Late Annealing            | Classifier boundary sharpening around equivocal cases      |              0.2812 |                 0.9654 |
| Epoch 8        |       7.4e-05  | Near-Terminal Convergence | OPTIMAL CONVERGENCE POINT (Peak Validation ROC-AUC)        |              0.2745 |                 0.9685 |
| Epoch 9        |       2e-05    | Terminal Fine-Tuning      | Loss plateauing; minimal parameter drift                   |              0.2731 |                 0.9678 |
| Epoch 10       |       5e-06    | Terminal Learning Floor   | FINAL MODEL CONVERGENCE (Minimum Loss = 0.2728)            |              0.2728 |                 0.9679 |