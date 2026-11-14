| Cache_Tier         |   Latency_ms | Speedup_vs_Cold   | Storage_Type        |
|:-------------------|-------------:|:------------------|:--------------------|
| Cold Disk Read     |      412.921 | 1.0x              | Raw Image File      |
| Disk Cache (.npy)  |       22.727 | 18.2x             | Pre-converted Numpy |
| Memory Cache (LRU) |        0.365 | 1131.6x           | RAM Buffer          |