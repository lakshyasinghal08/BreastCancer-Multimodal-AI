# Security & Data Governance Policy

## 1. Supported Versions
Security updates are actively applied to the production release:

| Release Version | Supported | Security Maintenance |
|---|---|---|
| `1.0.x` (Final Release) | Yes | Long-term institutional maintenance |
| `< 1.0.0` (Development RC) | No | Deprecated development versions |

## 2. Reporting Vulnerabilities
If you identify a security vulnerability—especially any potential data exposure risk, container isolation flaw, or dependency vulnerability—please report it privately to our security team:

- **Security Contact**: `security-advisory@consortium-research.org`
- Please provide details on the issue, steps to reproduce, and potential impact.
- We acknowledge reports within 48 hours and work with reporters on coordinated disclosure.

## 3. Patient Privacy & HIPAA Safe Harbor Standards
This repository strictly enforces patient de-identification according to the **HIPAA Safe Harbor standard (45 CFR 164.514)**:
- Zero real names, dates, hospital numbers, geographic locations, or facial biometric features are contained within code or documentation.
- All original clinical imaging and WSI biopsy files are excluded via `.gitignore` and must never be committed to version control.

