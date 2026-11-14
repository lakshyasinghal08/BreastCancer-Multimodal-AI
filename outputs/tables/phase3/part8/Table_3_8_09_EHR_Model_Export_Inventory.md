| Artifact Filename         | Format Specification         | Target Location     | Primary Utility                                   |
|:--------------------------|:-----------------------------|:--------------------|:--------------------------------------------------|
| best_model.pt             | Full PyTorch Checkpoint      | outputs/models/ehr/ | Complete Checkpoint with StateDict, Optim & Sched |
| ehr_model_best.pt         | PyTorch State Dict           | outputs/models/ehr/ | Model Weights for Multimodal Fusion & Inference   |
| ehr_model_final.pt        | PyTorch State Dict           | outputs/models/ehr/ | Epoch 35 Converged State Dict                     |
| ehr_model_torchscript.pt  | TorchScript Traced JIT       | outputs/models/ehr/ | C++ & Embedded LibTorch Runtime Execution         |
| ehr_model.onnx            | Open Neural Network Exchange | outputs/models/ehr/ | Hardware-Agnostic ONNX Runtime Deployment         |
| ehr_model_config.json     | JSON Configuration           | outputs/models/ehr/ | Model Architecture Hyperparameter Snapshot        |
| ehr_training_config.json  | JSON Configuration           | outputs/models/ehr/ | Training Pipeline & Optimization Registry         |
| ehr_training_history.json | JSON Telemetry               | outputs/models/ehr/ | Complete 35-Epoch Metrics & Trajectories          |
| ehr_metrics.json          | JSON Evaluation              | outputs/models/ehr/ | Independent Test Metrics & Bootstrap 95% CIs      |
| ehr_benchmark.json        | JSON Benchmark               | outputs/models/ehr/ | Latency, Throughput, and FLOPs Profiling          |