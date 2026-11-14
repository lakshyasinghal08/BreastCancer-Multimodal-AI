| Component              | Data Structure                     | Format                      | Purpose                           |
|:-----------------------|:-----------------------------------|:----------------------------|:----------------------------------|
| Model Parameters       | OrderedDict (state_dict)           | PyTorch Tensor Binary (.pt) | Restores Neural Weights & Biases  |
| Optimizer State        | state_dict (momentum, variance)    | PyTorch State Binary (.pt)  | Preserves Parameter Trajectory    |
| Scheduler State        | state_dict (step count, last_lr)   | PyTorch State Binary (.pt)  | Maintains Annealing Position      |
| AMP Scaler State       | GradScaler state_dict              | PyTorch State Binary (.pt)  | Preserves Dynamic Scale Factor    |
| Epoch & Step Counters  | Integer Values                     | Embedded & JSON Sidecar     | Ensures Exact Resumption Point    |
| Random Seeds & RNG     | torch.get_rng_state() + numpy      | Binary Tensor Tuple         | Bitwise Reproducibility Guarantee |
| Cryptographic Checksum | SHA-256 Digest (64 hex characters) | JSON Metadata Sidecar       | Prevents Bitrot & File Corruption |