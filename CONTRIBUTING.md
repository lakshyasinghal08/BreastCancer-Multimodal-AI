# Contributing Guidelines

Thank you for your interest in contributing to the **Multimodal Breast Cancer AI Framework**!

## 1. Scope & Research Freezing
Please note that all core scientific findings, trained checkpoints, and test cohort evaluations ($N=574$) reported in the primary manuscript are frozen. Contributions are primarily invited for:
- Inference engine optimizations and deployment packaging (e.g., Triton, TensorRT, CoreML).
- Documentation improvements, clinical case studies, and tutorials.
- Bug fixes in non-clinical utilities, testing harnesses, and tooling scripts.

## 2. Development Workflow
1. **Fork and Clone**: Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/inference-acceleration
   ```
2. **Environment Setup**: Ensure exact package versions match `requirements.txt`:
   ```bash
   conda env create -f environment.yml
   conda activate bca_multimodal
   ```
3. **Coding Standards**:
   - Write clean, type-hinted, and docstring-commented Python 3.10+ code.
   - Adhere to PEP 8 styling conventions.
   - Do NOT commit patient data, PHI, or large binaries ($> 50\text{ MB}$).
4. **Testing**: Run local regression and verification checks:
   ```bash
   python outputs/deployment/offline_inference_engine.py --device cpu
   ```
5. **Pull Request Protocol**: Submit your pull request with a clear description of changes, motivation, and test outputs.

## 3. Patient Privacy & Ethics
All contributors must ensure zero exposure of Protected Health Information (PHI). Do not upload proprietary hospital identifiers or raw clinical imaging files to issue trackers or pull requests.

