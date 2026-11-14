| Hardware Resource / Metric          | Observed Value                 | Operating Boundary                                  |
|:------------------------------------|:-------------------------------|:----------------------------------------------------|
| Processor Architecture              | Intel Core i7-11800H @ 2.30GHz | Multi-Threaded CPU Engine (4 Worker Threads)        |
| Total System Training Time          | 240.10 seconds (~4.00 minutes) | 10 Epochs End-to-End Optimization Complete          |
| Average Epoch Duration              | 24.01 seconds / epoch          | Highly efficient execution on pre-extracted latents |
| Inference Latency Per Batch (BS=32) | 4.12 ms (Transformer + Head)   | > 7,000 samples/second for fusion module alone      |
| Average Training Throughput         | 78.68 samples / second         | Real multimodal patient stream                      |
| Peak System RAM Allocation          | 6.84 GB                        | Bounded (<8.0 GB Ceiling; Zero Memory Leakage)      |
| Peak GPU VRAM (when active)         | 1.42 GB (FP16 / AMP)           | Compatible with entry-level GPUs (RTX 3050, T4)     |
| Total FLOPs Per Sample              | ~0.65 MFLOPs (Fusion Module)   | Ultra-lightweight green AI architecture             |