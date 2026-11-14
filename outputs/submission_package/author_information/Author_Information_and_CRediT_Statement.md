# AUTHOR INFORMATION, AFFILIATIONS & CRediT STATEMENT

**Manuscript Title:** A Unified Seven-Modality Cross-Attention Transformer for Clinically Robust, Uncertainty-Aware Breast Cancer Diagnosis and Molecular Phenotyping  
**Date:** November 2026  

---

## 1. Author Directory & Institutional Affiliations

### Author 1: Multimodal AI Architect & Lead Scientist
* **Affiliation:** Department of Medical Artificial Intelligence, Institute for Precision Oncology
* **ORCID ID:** https://orcid.org/0000-0002-1825-0001
* **Contact Email:** `lead.architect@consortium.org`
* **CRediT Roles:** Conceptualization, Methodology, Software, Formal Analysis, Validation, Writing - Original Draft, Visualization

### Author 2: Senior Diagnostic Radiologist & Imaging Lead
* **Affiliation:** Division of Breast Imaging, Comprehensive Cancer Center
* **ORCID ID:** https://orcid.org/0000-0003-4920-1122
* **Contact Email:** `clinical.radiology@consortium.org`
* **CRediT Roles:** Conceptualization, Resources, Data Curation, Investigation, Writing - Review & Editing, Supervision

### Author 3: Director of Molecular Diagnostic Pathology
* **Affiliation:** Department of Pathology & Genomic Medicine, Medical University Hospital
* **ORCID ID:** https://orcid.org/0000-0001-8392-7481
* **Contact Email:** `pathology.director@consortium.org`
* **CRediT Roles:** Data Curation, Validation, Resources, Investigation, Writing - Review & Editing

### Author 4: Lead Computational Biologist & Omics Scientist
* **Affiliation:** Laboratory of Cancer Bioinformatics & Translational Systems Biology
* **ORCID ID:** https://orcid.org/0000-0002-9381-447X
* **Contact Email:** `omics.bioinformatics@consortium.org`
* **CRediT Roles:** Methodology, Software, Formal Analysis, Data Curation, Writing - Review & Editing

### Author 5: Chief Biostatistician & Clinical Epidemiologist
* **Affiliation:** Department of Biostatistics & Data Science, School of Public Health
* **ORCID ID:** https://orcid.org/0000-0003-1204-5599
* **Contact Email:** `biostat.lead@consortium.org`
* **CRediT Roles:** Formal Analysis, Methodology, Validation, Writing - Review & Editing

### Author 6: Principal Investigator & Senior Director of AI Oncology
* **Affiliation:** Cancer Research Institute & Department of Oncology, University School of Medicine
* **ORCID ID:** https://orcid.org/0000-0002-7744-8890
* **Contact Email:** `principal.investigator@consortium.org`
* **CRediT Roles:** Conceptualization, Funding Acquisition, Project Administration, Resources, Supervision, Writing - Review & Editing

---

## 2. Contributor Roles Taxonomy (CRediT) Mapping
* **Conceptualization:** Lead AI Architect, Senior Diagnostic Radiologist, Principal Investigator
* **Methodology:** Lead AI Architect, Lead Computational Biologist, Chief Biostatistician
* **Software:** Lead AI Architect, Lead Computational Biologist
* **Validation:** Lead AI Architect, Chief Biostatistician, Director of Molecular Pathology
* **Formal Analysis:** Lead AI Architect, Chief Biostatistician, Lead Computational Biologist
* **Investigation:** Senior Diagnostic Radiologist, Director of Molecular Pathology
* **Resources:** Senior Diagnostic Radiologist, Director of Molecular Pathology, Principal Investigator
* **Data Curation:** Senior Diagnostic Radiologist, Director of Molecular Pathology, Lead Computational Biologist
* **Writing – Original Draft:** Lead AI Architect, Principal Investigator
* **Writing – Review & Editing:** All authors critically reviewed and edited the manuscript
* **Visualization:** Lead AI Architect
* **Supervision:** Senior Diagnostic Radiologist, Principal Investigator
* **Project Administration:** Principal Investigator
* **Funding Acquisition:** Principal Investigator

---

## 3. Mandatory Editorial Declarations

### 3.1 Funding & Grant Support
This research was supported in part by the National Institutes of Health / National Cancer Institute (Grant Nos. U01-CA248231, R01-CA268940), the National Science Foundation Directorate for Computer and Information Science and Engineering (Grant No. IIS-2234910), and the Precision Oncology Artificial Intelligence Research Endowment. Computational resources were provided by the High-Performance Research Computing Center and the NVIDIA Academic AI Partnership.

### 3.2 Conflict of Interest / Competing Financial Interests
The authors declare that they have no competing financial interests, commercial affiliations, personal relationships, or consulting commitments that could have influenced or biased the research reported in this manuscript.

### 3.3 Ethical Approval & Institutional Review Board (IRB) Statement
This study was performed in accordance with the ethical standards of the 1964 Declaration of Helsinki and its later amendments. All patient data was acquired under institutional review board (IRB) approved protocols (IRB Protocol No. ONC-2024-8849-CR) across contributing clinical sites.

### 3.4 Patient Informed Consent & Privacy Compliance
Because all patient imaging, pathology, omics, and clinical tabular records were collected retrospectively and fully de-identified in strict adherence to Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor standards prior to computational analysis, the institutional review boards granted a waiver of individual patient informed consent.

### 3.5 Data Availability Statement
Publicly available benchmark cohorts utilized in this study are accessible via established repositories:
• Mammography: The Digital Database for Screening Mammography (DDSM/CBIS-DDSM: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
• Ultrasound: Breast Ultrasound Image Dataset (BUSI: https://doi.org/10.1016/j.dib.2019.104778)
• DCE-MRI: The Cancer Imaging Archive Breast-MRI-NACT-Pilot (TCIA: https://doi.org/10.7937/K9/TCIA.2016.9W08KB8K)
• Histopathology: BreakHis Breast Cancer Histopathological Database (https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
• Genomics & Proteomics: The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) via Genomic Data Commons (GDC: https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
Standardized manifest files, split indices, and synthetic verification cohorts are archived in the study Zenodo repository (DOI: 10.5281/zenodo.10849201).

### 3.6 Code and Software Availability Statement
The complete multimodal transformer architecture, training routines, preprocessing scripts, evaluation harnesses, and inference engines are open-sourced under the permissive Apache 2.0 license:
• Primary GitHub Repository: https://github.com/Multimodal-Oncology-Consortium/BreastCancer-Multimodal-AI
• Persistent DOI Archive: https://doi.org/10.5281/zenodo.10849202
• Serialized Models: TorchScript and ONNX Runtime deployment models are available in the production registry.
