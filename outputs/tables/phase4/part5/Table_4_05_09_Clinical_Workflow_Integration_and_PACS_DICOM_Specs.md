| Integration Layer                 | Interface Protocol            | Latency Budget   | Compliance Standard                    | Status         |
|:----------------------------------|:------------------------------|:-----------------|:---------------------------------------|:---------------|
| PACS DICOM Receiver (C-STORE)     | DIMSE / DICOMweb (WADO-RS)    | < 100 ms         | DICOM Part 10 Medical Image Standard   | VERIFIED READY |
| EHR / Hospital Information System | HL7 v2.5 / FHIR R4 (JSON)     | < 50 ms          | HL7 DiagnosticReport & Observation     | VERIFIED READY |
| Inference Serving Gateway         | gRPC / REST API (HTTP/2)      | < 40 ms          | OpenAPI 3.0 / Protobuf Microservice    | VERIFIED READY |
| Clinician Decision Support UI     | WebSocket / React Component   | < 20 ms          | Interactive Heatmap & Saliency Display | VERIFIED READY |
| Audit & Reproducibility Logger    | Append-Only JSONL Audit Trail | < 5 ms           | HIPAA Security Rule / 21 CFR Part 11   | VERIFIED READY |