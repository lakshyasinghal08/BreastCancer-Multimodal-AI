"""Publication Table Exporter for Medical Dataset Validation.

Exports structured research tables across 4 standard academic formats:
CSV, Excel (.xlsx), GitHub Markdown (.md), and LaTeX (.tex).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger("BreastCancer-Multimodal-AI")


def export_publication_table(
    df: pd.DataFrame,
    table_name: str,
    tables_dir: Path,
    reports_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """Exports a single pandas DataFrame into CSV, Excel, Markdown, and LaTeX.

    Args:
        df: DataFrame to export.
        table_name: Base filename identifier (e.g., 'dataset_integrity_summary').
        tables_dir: Destination folder for tabular data (outputs/tables/).
        reports_dir: Destination folder for academic reports/LaTeX (outputs/reports/).
        logger: Logger instance.

    Returns:
        Dictionary mapping format ('csv', 'xlsx', 'md', 'tex') to output Path.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    # 1. Export CSV
    csv_p = tables_dir / f"{table_name}.csv"
    df.to_csv(csv_p, index=False)
    exported["csv"] = csv_p

    # 2. Export Excel (.xlsx) using xlsxwriter
    xlsx_p = tables_dir / f"{table_name}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_p, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=table_name[:31], index=False)
    except Exception:
        with pd.ExcelWriter(xlsx_p) as writer:
            df.to_excel(writer, sheet_name=table_name[:31], index=False)
    exported["xlsx"] = xlsx_p

    # 3. Export Markdown
    md_p = reports_dir / f"{table_name}.md"
    md_content = f"# Table: {table_name.replace('_', ' ').title()}\n\n" + df.to_markdown(index=False) + "\n"
    md_p.write_text(md_content, encoding="utf-8")
    exported["md"] = md_p

    # 4. Export LaTeX (.tex)
    tex_p = reports_dir / f"{table_name}.tex"
    tex_content = df.to_latex(index=False, caption=table_name.replace('_', ' ').title(), label=f"tab:{table_name}")
    tex_p.write_text(tex_content, encoding="utf-8")
    exported["tex"] = tex_p

    return exported


def export_all_validation_tables(
    tables_dict: Dict[str, pd.DataFrame],
    tables_dir: Path,
    reports_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Path]]:
    """Exports all 6 publication validation tables into CSV, Excel, Markdown, and LaTeX formats."""
    logger = logger or logging.getLogger("BreastCancer-Multimodal-AI")
    results = {}

    # Also build a single combined multi-sheet workbook
    combined_xlsx = tables_dir / "all_validation_tables.xlsx"
    try:
        with pd.ExcelWriter(combined_xlsx, engine="xlsxwriter") as writer:
            for name, df in tables_dict.items():
                sheet_title = name[:30].replace("_", " ").title()
                df.to_excel(writer, sheet_name=sheet_title, index=False)
    except Exception:
        with pd.ExcelWriter(combined_xlsx) as writer:
            for name, df in tables_dict.items():
                sheet_title = name[:30].replace("_", " ").title()
                df.to_excel(writer, sheet_name=sheet_title, index=False)

    for name, df in tables_dict.items():
        results[name] = export_publication_table(df, name, tables_dir, reports_dir, logger)

    logger.info(f"Exported {len(tables_dict)} publication tables across CSV, Excel, Markdown, and LaTeX formats.")
    return results
