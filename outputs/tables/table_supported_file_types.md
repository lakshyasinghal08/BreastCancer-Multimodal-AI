# Comprehensive File Type Support and Data Ingestion Protocols

| Modality       | File_Format   | Primary_Reader        | Lazy_Support              | Error_Handling   |
|:---------------|:--------------|:----------------------|:--------------------------|:-----------------|
| Mammography    | .dcm / .jpg   | pydicom / PIL         | Yes (On-demand decode)    | Graceful Skip    |
| Ultrasound     | .png          | PIL Image             | Yes (Metadata First)      | Graceful Skip    |
| Histopathology | .png / .svs   | PIL Image / OpenSlide | Yes (Tile-level Lazy)     | Graceful Skip    |
| MRI            | .png          | PIL Image             | Yes (Slice-level Lazy)    | Graceful Skip    |
| Clinical       | .csv          | pandas (chunked)      | Yes (Column Filtering)    | Graceful Skip    |
| EHR            | .csv          | pandas (chunked)      | Yes (Column Filtering)    | Graceful Skip    |
| Genomic        | .csv          | pandas (indexed)      | Yes (Sample Matrix Slice) | Graceful Skip    |
