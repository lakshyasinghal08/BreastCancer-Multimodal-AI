| Variable Name            | Unit               | Sampling Rate         | Target Level     | Diagnostics                       |
|:-------------------------|:-------------------|:----------------------|:-----------------|:----------------------------------|
| train_loss               | Cross-Entropy Loss | Per Batch & Per Epoch | < 0.350          | Optimization Convergence          |
| val_loss                 | Cross-Entropy Loss | Per Epoch             | < 0.400          | Generalization Gap Tracking       |
| val_roc_auc              | AUC (0.0 to 1.0)   | Per Epoch             | > 0.850          | Primary Discriminative Quality    |
| mean_grad_norm           | L2 Norm            | Per Batch             | <= 1.0 (Clipped) | Numerical Stability Guarantee     |
| throughput_samples_per_s | Patients / Second  | Per Epoch             | > 8.0 spl/s      | Computational Pipeline Efficiency |
| ram_used_gb              | Gigabytes (GB)     | Per Epoch             | < 12.0 GB        | Memory Leakage Prevention         |
| learning_rate            | Step Rate (float)  | Per Epoch             | 1e-6 to 1e-3     | Annealing Schedule Fidelity       |