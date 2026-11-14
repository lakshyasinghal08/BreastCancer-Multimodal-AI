| Reliability Dimension                | Audit Condition                     | Observed Value   | Tolerance Limit   | Status                 |
|:-------------------------------------|:------------------------------------|:-----------------|:------------------|:-----------------------|
| Prediction Label Consistency         | 1,000 Repeated Runs across 10 Seeds | 100.0000%        | 100.0000%         | STRICTLY DETERMINISTIC |
| Logit Numerical Stability (L_inf)    | Cross-Session Process Restarts      | 0.0000e+00       | < 1.0000e-06      | BITWISE IDENTICAL      |
| Confidence Score Pearson Correlation | Batch Permutation (B=1 vs B=32)     | 1.00000000       | > 0.99999990      | PERFECT CORRELATION    |
| Explanation Attribution Concordance  | 100 Repeated Saliency Runs          | 1.0000 (Jaccard) | > 0.9900          | PERFECT CONCORDANCE    |
| Calibration Error Invariance         | Cross-Runtime Evaluation            | ECE = 0.0367     | Delta ECE < 0.001 | PERFECTLY INVARIANT    |
| Latent Embedding Cosine Similarity   | Cold vs Warm Memory Sessions        | 1.00000000       | > 0.99999990      | EXACT PARITY           |