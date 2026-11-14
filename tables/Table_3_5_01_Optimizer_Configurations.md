| Optimizer         |   Learning Rate | Betas / Momentum   |   Weight Decay | Epsilon   | Parameter Grouping              | Decay Exclusion   |
|:------------------|----------------:|:-------------------|---------------:|:----------|:--------------------------------|:------------------|
| AdamW (Default)   |           0.001 | (0.9, 0.999)       |         0.0001 | 1e-8      | Differential (0.5x, 1.0x, 1.5x) | Bias & Norm       |
| Adam              |           0.001 | (0.9, 0.999)       |         0.0001 | 1e-8      | Uniform / Differential          | Optional          |
| SGD with Momentum |           0.01  | 0.90 (Nesterov)    |         0.0001 | N/A       | Supported                       | Bias & Norm       |
| RMSprop           |           0.001 | 0.99 (Alpha)       |         0.0001 | 1e-8      | Supported                       | Bias & Norm       |