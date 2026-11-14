# Long-Term Preservation & Data Governance Strategy (5-Year Plan)

## 1. Compliance with FAIR Data Principles
- **Findable**: All datasets, weights, tables, and scripts are cataloged with standardized semantic filenames and SHA-256 identifiers in `artifact_catalog.json`.
- **Accessible**: Stored in open, non-proprietary formats (CSV, OpenXML XLSX, Markdown, LaTeX, JSON, PNG, SVG, PDF, EPS, ONNX).
- **Interoperable**: Adheres to standard DICOM PS3.x, HL7/FHIR, and W3C PROV-O data interchange formats.
- **Reusable**: Licensed under permissive research usage with complete dependency lockfiles (`requirements.txt`).

## 2. Institutional Bit-Rot Protection & Hash Auditing
All frozen artifacts are hashed with SHA-256 and stored in `outputs/artifact_inventory/checksum_manifest.json`. An automated annual audit verifies that no bit-flips or file corruptions have occurred across institutional optical / cold storage archives.
