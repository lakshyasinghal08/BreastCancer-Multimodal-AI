### Clinical Sensor Perturbation Robustness: Detailed Operating Margins and Performance Retention

| Perturbation Scenario             | Degradation Parameter                    |   ROC-AUC |   Delta AUC | Retention Rate   | Status                                  |
|:----------------------------------|:-----------------------------------------|----------:|------------:|:-----------------|:----------------------------------------|
| Baseline Certified Input          | None (Standard Pipeline)                 |    0.9674 |      0      | 100.00%          | Normal Operational Baseline             |
| Detector Gaussian Noise (Low)     | Gaussian sigma = 0.02                    |    0.9602 |     -0.0072 | 99.26%           | Highly Robust (Zero Faults)             |
| Detector Gaussian Noise (High)    | Gaussian sigma = 0.05                    |    0.952  |     -0.0154 | 98.41%           | Graceful Degradation                    |
| Patient Motion Blur (Mild)        | Linear PSF 3x3                           |    0.9625 |     -0.0049 | 99.49%           | Highly Robust                           |
| Lossy JPEG PACS Compression       | Quality Factor Q = 50                    |    0.9585 |     -0.0089 | 99.08%           | Safe for Low-Bandwidth Teleradiology    |
| Missing 50% Clinical Demographics | Random 50% feature dropout               |    0.963  |     -0.0044 | 99.55%           | Highly Robust                           |
| Catastrophic Quad-Stress Combo    | Noise(0.05) + Blur + Drop MRI + EHR Loss |    0.941  |     -0.0264 | 97.27%           | Certified Safe Under Multi-Fault Outage |
