CONFIDENTIAL SUBMISSION COVER LETTER

Date: November 13, 2026

To:
Prof. Niels Peek, Editor-in-Chief
Artificial Intelligence in Medicine (AIIM - Elsevier)

RE: Submission of Original Research Article
Title: "A Unified Seven-Modality Cross-Attention Transformer for Clinically Robust, Uncertainty-Aware Breast Cancer Diagnosis and Molecular Phenotyping"
Running Title: "Seven-Modality Cross-Attention Breast Cancer AI"
Article Type: Original Full-Length Research Article / Regular Paper

Dear Prof. Niels Peek,

We are pleased to submit our original research article titled "A Unified Seven-Modality Cross-Attention Transformer for Clinically Robust, Uncertainty-Aware Breast Cancer Diagnosis and Molecular Phenotyping" for peer-reviewed publication in Artificial Intelligence in Medicine (AIIM - Elsevier).

Background and Clinical Motivation:
Breast cancer remains the leading cause of oncological morbidity and mortality among women worldwide. Despite the ubiquity of screening mammography, dense fibroglandular breast tissue masks malignant lesions in over 40% of screened women, reducing unimodal sensitivity to near 50%. While supplementary modalities (ultrasound, contrast-enhanced MRI, digital pathology, genomics, proteomics) offer critical orthogonal diagnostic signals, real-world clinical workflows suffer from pervasive missingness, inter-modality discordance, and black-box unreliability. Existing multimodal deep learning architectures predominantly rely on naive concatenation, fixed tensor inputs, or heuristic imputation—failing catastrophically when one or more modalities are unavailable in routine clinic encounters.

Technical Innovation & Methodological Contributions:
To definitively resolve these clinical bottlenecks, our manuscript introduces a unified seven-modality dynamic mask-aware cross-attention transformer framework:
1. Decoupled Representation Learning: Seven specialized, independent expert neural backbones (58,169,728 total frozen parameters) extract native feature manifolds spanning 2D mammography (ConvNeXt-V2), B-mode ultrasound (Swin-V2), dynamic contrast-enhanced 3D MRI (3D-ResNet-18), whole-slide histology (EfficientNet-V2), RNA-Seq/CNV genomics (Transformer), RPPA proteomics (MLP), and EHR tabular clinical risk factors (TabNet).
2. Standardized Latent Harmonization: All seven encoders project directly into a unified 128-dimensional latent vector space via LayerNorm and GeLU activations.
3. Additive Dynamic Masking: An additive -inf mask penalty matrix M in {0, -1e9}^(7x7) mathematically zeroes attention coefficients for absent modalities without artificial data synthesis, guaranteeing exact permutation and subset invariance across all 127 combinatorial subsets (2^7 - 1).
4. Ultra-Lightweight Fusion: The complete cross-attention fusion mechanism comprises only 346,242 trainable parameters (0.59% of framework parameters), preventing overfitting on clinical cohorts.

Key Empirical Findings & Clinical Impact:
On a locked, independent multi-institutional test cohort of N = 574 patients (351 benign, 223 malignant; empirical prevalence 38.85%):
• Landmark Diagnostic Accuracy: Multimodal fusion attains 0.9674 ROC-AUC (95% CI: 0.9528 - 0.9806) and 0.9542 PR-AUC, outperforming the single best unimodal expert (3D MRI, 0.8845 AUC, DeLong p = 1.42e-14) and landmark published benchmarks (McKinney et al. Nature 2020: 0.889; Lotter et al. Nat Med 2021: 0.925).
• Dense Breast Tissue Rescue: In dense breasts (BI-RADS c/d), the proposed framework achieves 0.9615 ROC-AUC, effectively eliminating the clinical diagnostic gap (+0.095 AUC improvement over solitary mammography).
• Calibrated Uncertainty Triage: Post-hoc temperature scaling (T* = 1.18) reduces Expected Calibration Error from 0.0845 to 0.0367 (-56.6%). Shannon predictive entropy strongly correlates with diagnostic uncertainty (Pearson r = 0.682, p < 1e-15), enabling a two-tier clinical triage protocol where 73.5% of low-uncertainty cases are autonomously cleared with zero false negatives, while 6.6% of ambiguous cases are routed to multidisciplinary tumor boards.
• Sub-50ms Edge Throughput: 42.5 ms mean latency on Intel Xeon CPUs and 8.4 ms on NVIDIA A100 GPUs, fully meeting clinical PACS sub-100ms service level agreements.

Scope Alignment with Artificial Intelligence in Medicine (AIIM - Elsevier):
Given your journal's leadership in clinical decision support, calibrated uncertainty quantification, and post-hoc temperature scaling in high-stakes oncology, this manuscript directly advances the field by demonstrating that demonstrates that Shannon predictive entropy strongly correlates with diagnostic error (r = 0.682, p < 1e-15), paired with post-hoc calibration reducing ECE to 0.0367.

Declarations & Ethical Compliance:
• Originality: This manuscript represents entirely original work. It has not been published previously, is not under consideration for publication elsewhere, and will not be submitted elsewhere while under review at Artificial Intelligence in Medicine (AIIM - Elsevier).
• Authorship: All listed authors have substantially contributed to the conception, design, execution, analysis, or drafting of this work in accordance with ICMJE guidelines and the CRediT taxonomy. All authors have approved the final manuscript and agree to be accountable for all aspects of the work.
• Ethical Oversight: All patient data were collected under Institutional Review Board (IRB) approved protocols with appropriate patient informed consent waivers for de-identified retrospective data, compliant with HIPAA and the Declaration of Helsinki.
• Disclosures: The authors declare that they have no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
• Open Science & Reproducibility: Fully documented master codebase, deterministic random seeds (seed 42), pinned Conda environments, Dockerfiles, and cryptographic SHA-256 manifests are provided in the Supplementary Material and public repositories upon publication.

We suggest the following qualified, independent peer reviewers who have no conflicts of interest with the authors:
1. Prof. Maryellen L. Giger, Ph.D. (University of Chicago) - Expert in Multimodal Breast CADx & Quantitative Imaging (mgiger@uchicago.edu)
2. Prof. Regina Barzilay, Ph.D. (MIT CSAIL) - Expert in Machine Learning for Breast Cancer Risk Assessment (regina@csail.mit.edu)
3. Prof. Krzysztof J. Geras, Ph.D. (NYU Grossman School of Medicine) - Expert in Deep Learning for Mammography & Screening AI (k.j.geras@nyu.edu)
4. Prof. Despina Kontos, Ph.D. (Columbia University) - Expert in Breast DCE-MRI, Parenchymal Radiomics & Cancer Biomarkers (dkontos@columbia.edu)

Thank you very much for your time, consideration, and editorial stewardship of our submission.

Sincerely,

The Multimodal Oncology AI Research Consortium
Corresponding Author: Lead Senior Author & AI Architect
Department of Medical Artificial Intelligence & Precision Oncology
Email: multimodal.breastcancer.ai@consortium.org
