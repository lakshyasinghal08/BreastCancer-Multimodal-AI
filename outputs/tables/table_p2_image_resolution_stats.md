| Modality                   | Native_Resolution_Range   |   Native_Aspect_Ratio | Preprocessed_Resolution   |   Channels | Interpolation   | Padding         |
|:---------------------------|:--------------------------|----------------------:|:--------------------------|-----------:|:----------------|:----------------|
| Mammography (CBIS-DDSM)    | 2500x3500 to 4500x6000    |                  0.71 | 512 x 512                 |          1 | Bilinear        | Letterbox Zero  |
| Ultrasound (BUSI)          | 400x400 to 760x600        |                  1.25 | 256 x 256                 |          1 | Bilinear        | Letterbox Zero  |
| Histopathology (BreakHis)  | 700 x 460                 |                  1.52 | 224 x 224                 |          3 | Bilinear        | Letterbox Zero  |
| Histopathology (IDC Tiles) | 50 x 50                   |                  1    | 224 x 224                 |          3 | Bilinear        | Bicubic Upscale |
| Breast DCE-MRI             | 256 x 256                 |                  1    | 256 x 256                 |          1 | Bilinear        | Native Square   |