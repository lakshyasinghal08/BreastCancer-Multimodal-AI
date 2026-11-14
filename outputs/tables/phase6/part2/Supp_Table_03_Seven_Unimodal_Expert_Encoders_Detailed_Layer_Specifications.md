### Seven Unimodal Expert Encoders Detailed Layer Specifications, Tensor Dimensions and FLOPs

| Expert Model            | Backbone Architecture   |   Total Layers | Input Dimensions     |   Output Feature Dim | Projection Layer               |   FLOPs (G) | Parameters   |
|:------------------------|:------------------------|---------------:|:---------------------|---------------------:|:-------------------------------|------------:|:-------------|
| Mammography Expert      | ResNet-50               |             50 | (B, 1, 224, 224)     |                 2048 | Linear(2048, 128) -> LayerNorm |        4.12 | 23.51 M      |
| Ultrasound Expert       | EfficientNet-B0         |             81 | (B, 3, 224, 224)     |                 1280 | Linear(1280, 128) -> LayerNorm |        0.39 | 4.01 M       |
| Histopathology Expert   | DenseNet-121            |            121 | (B, 3, 224, 224)     |                 1024 | Linear(1024, 128) -> LayerNorm |        2.88 | 6.95 M       |
| Breast MRI Expert       | 3D ResNet-18            |             18 | (B, 1, 16, 112, 112) |                  512 | Linear(512, 128) -> LayerNorm  |        7.45 | 11.17 M      |
| Clinical Tabular Expert | Residual MLP            |              4 | (B, 30)              |                  256 | Linear(256, 128) -> LayerNorm  |        0.01 | 0.18 M       |
| Longitudinal EHR Expert | Temporal BiGRU          |              3 | (B, 45)              |                  256 | Linear(256, 128) -> LayerNorm  |        0.02 | 0.24 M       |
| Genomics Expert         | Pathway MLP             |              5 | (B, 500)             |                  512 | Linear(512, 128) -> LayerNorm  |        0.03 | 0.62 M       |
