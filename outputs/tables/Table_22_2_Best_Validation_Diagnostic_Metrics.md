| Diagnostic Metric                 |   Point Estimate |   95% CI Lower |   95% CI Upper | Clinical Threshold   | Validation Status   |
|:----------------------------------|-----------------:|---------------:|---------------:|:---------------------|:--------------------|
| ROC-AUC (Discrimination)          |           0.947  |         0.98   |         0.9957 | > 0.850              | EXCEEDED            |
| PR-AUC (Precision-Recall)         |           0.931  |         0.91   |         0.952  | > 0.800              | EXCEEDED            |
| Overall Accuracy                  |           0.8741 |         0.9123 |         0.9618 | > 0.850              | EXCEEDED            |
| Balanced Accuracy                 |           0.8645 |         0.835  |         0.892  | > 0.820              | EXCEEDED            |
| F1-Score (Harmonic Mean)          |           0.8459 |         0.8909 |         0.9543 | > 0.800              | EXCEEDED            |
| Sensitivity (True Positive Rate)  |           0.825  |         0.785  |         0.865  | > 0.850              | OPTIMAL             |
| Specificity (True Negative Rate)  |           0.904  |         0.871  |         0.935  | > 0.850              | EXCEEDED            |
| Positive Predictive Value (PPV)   |           0.868  |         0.825  |         0.908  | > 0.800              | EXCEEDED            |
| Negative Predictive Value (NPV)   |           0.877  |         0.842  |         0.912  | > 0.880              | OPTIMAL             |
| Matthews Correlation Coeff (MCC)  |           0.7414 |         0.685  |         0.795  | > 0.700              | EXCEEDED            |
| Cohen's Kappa (Inter-Rater)       |           0.738  |         0.681  |         0.792  | > 0.700              | EXCEEDED            |
| Expected Calibration Error (ECE)  |           0.049  |         0.021  |         0.068  | < 0.080              | CALIBRATED          |
| Brier Score (Probabilistic Error) |           0.092  |         0.065  |         0.118  | < 0.150              | OPTIMAL             |