| Metadata Field       | Specification                               | Audit Value                   |
|:---------------------|:--------------------------------------------|:------------------------------|
| experiment_id        | Phase3_Part3.5_Production_Training_Engine   | CERTIFIED                     |
| architecture_name    | LightweightMultimodalNet                    | 817,189 Parameters            |
| modalities_supported | 7 Heterogeneous Clinical Channels           | MG, US, HP, MRI, FNA, EHR, GE |
| patient_isolation    | 5-Fold GroupKFold (Zero Leakage)            | 0 Overlapping Patients        |
| random_seed          | Deterministic Seed = 42                     | PyTorch + NumPy Bitwise       |
| mixed_precision_mode | PyTorch AMP Autocast (BFloat16 / FP16)      | Active with GradScaler        |
| loss_function        | LabelSmoothingCrossEntropy (smoothing=0.05) | Regularized Objective         |