| Config File               | Scope                      | Key Parameters                                   | Hash (SHA256)   |
|:--------------------------|:---------------------------|:-------------------------------------------------|:----------------|
| pipeline_config.json      | End-to-End Orchestration   | batch_size=32, num_workers=4, pin_memory=True    | a7f3c91b8d2e... |
| split_config.json         | Patient Partitioning       | train=0.70, val=0.15, test=0.15, seed=42         | b4e2d19f0a5c... |
| cache_config.json         | Multi-Tier Caching         | mem_cap=1000, disk_fmt=npy, eviction=LRU         | 9c1a5f82d3e0... |
| augmentation_config.json  | Clinical Transformations   | 18 transforms, enabled=False (baseline)          | e3b8a1c92d4f... |
| dataloader_config.json    | Worker & Collate Engine    | prefetch=2, persistent=True, dynamic_batch=True  | 7d2e9f1a5b8c... |
| benchmark_config.json     | Stress Testing Engine      | bs_range=[1,64], workers=[0,8], timeout=120s     | 6f0b4d9a1e3c... |
| dataset_class_config.json | Multimodal Class Contracts | 8 classes, lazy_load=True, error_policy=fallback | 5e1c8d2a9f4b... |