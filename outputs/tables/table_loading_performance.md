# Summary of Cold vs Cached Data Loading Latency and Throughput

| Access_Mode            | Mean_Latency_ms   | Median_Latency_ms   | P95_Latency_ms   |   Throughput_Smp_Sec |
|:-----------------------|:------------------|:--------------------|:-----------------|---------------------:|
| Cold Load (Disk Read)  | 2.08 ms           | 2.92 ms             | 4.25 ms          |            43018.2   |
| Warm Load (LRU Cached) | 0.001 ms          | 0.001 ms            | 0.002 ms         |                1e+06 |
