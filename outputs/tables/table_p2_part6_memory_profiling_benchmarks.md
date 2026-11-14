# Pipeline Memory Profiling & Resource Utilization Audit

| Subsystem                     |   Allocated_RAM_MB |   RSS_MB |   Disk_IO_MBs |   CPU_Utilization_Pct | Memory_Leakage_Rate   |
|:------------------------------|-------------------:|---------:|--------------:|----------------------:|:----------------------|
| Raw Dataset Ingestion         |              120.4 |    720.5 |          42.1 |                  18.5 | 0.00 MB/hr            |
| Image Preprocessing           |              185.2 |    745.2 |          58.4 |                  34.2 | 0.00 MB/hr            |
| Tabular Scalers               |               45   |    750.1 |          12   |                   8.1 | 0.00 MB/hr            |
| LRU In-Memory Cache           |               50   |    780   |           0   |                   2   | 0.00 MB/hr            |
| Batch Collation & Tensor Feed |               30   |    810   |           5.2 |                  12.4 | 0.00 MB/hr            |
