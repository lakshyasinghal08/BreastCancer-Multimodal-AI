| Benchmark Metric             | Execution Mode             | Measured Value                |
|:-----------------------------|:---------------------------|:------------------------------|
| Batch Size 1 Step Latency    | Single Sample CPU/GPU      | 1.210 ms (P99: 2.769 ms)      |
| Batch Size 1 Throughput      | Single Sample CPU/GPU      | 826.4 samples/sec             |
| Batch Size 16 Step Latency   | Batched Processing         | 1.159 ms                      |
| Batch Size 16 Throughput     | Batched Processing         | 13801.8 samples/sec           |
| Batch Size 32 Step Latency   | Batched Processing         | 1.234 ms                      |
| Batch Size 32 Throughput     | Batched Processing         | 25928.6 samples/sec           |
| Total Parameter Footprint    | In-Memory Float32          | 43,490 parameters (169.88 KB) |
| Theoretical FLOPs per Sample | Forward Pass Multiply-Adds | 63,232 FLOPs (~0.05 MFLOPs)   |