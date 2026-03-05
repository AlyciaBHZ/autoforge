"""
Automated LaTeX paper generation pipeline for AutoForge AI research agent framework.

This module orchestrates the generation of publication-quality academic papers from
research findings, including automatic section writing, figure/table generation,
bibliography management, and PDF compilation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# LaTeX Template Management
# ══════════════════════════════════════════════════════════════


class TemplateEngine:
    """Template-driven LaTeX paper generation."""

    TEMPLATE_PREAMBLE = {
        "neurips": r"""\documentclass{article}
\usepackage[final]{nips_2024}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subfigure}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\title{${TITLE}}
\author{${AUTHORS}}
\begin{document}
\maketitle
""",
        "icml": r"""\documentclass[11pt]{article}
\usepackage{icml2024}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{subfigure}
\usepackage{booktabs}
\icmltitlerunning{${TITLE}}
\begin{document}
\twocolumn[
\icmltitle{${TITLE}}
\icmlsetsummaryauthor{${FIRST_AUTHOR}}
]
\begin{abstract}
${ABSTRACT}
\end{abstract}
""",
        "iclr": r"""\documentclass{article}
\usepackage{iclr2024_conference}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{subcaption}
\usepackage{booktabs}
\iclrfinalcopy
\title{${TITLE}}
\author{${AUTHORS}}
\begin{document}
\maketitle
\begin{abstract}
${ABSTRACT}
\end{abstract}
""",
        "arxiv": r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\title{${TITLE}}
\author{${AUTHORS}}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
${ABSTRACT}
\end{abstract}
""",
    }

    def __init__(self, template_name: str = "arxiv"):
        """Initialize template engine."""
        if template_name not in self.TEMPLATE_PREAMBLE:
            logger.warning(f"Unknown template {template_name}, using arxiv")
            template_name = "arxiv"
        self.template_name = template_name

    def render_preamble(
        self,
        title: str,
        authors: list[dict],
        abstract: str = "",
    ) -> str:
        """Render document preamble."""
        template = self.TEMPLATE_PREAMBLE[self.template_name]
        authors_str = ", ".join(
            f"{a['name']}" + (f"$^{{{a.get('affiliation_id', '')}}}$" if a.get('affiliation_id') else "")
            for a in authors
        )
        first_author = authors[0]['name'] if authors else "Anonymous"

        result = template
        result = result.replace("${TITLE}", title.replace("&", r"\&"))
        result = result.replace("${AUTHORS}", authors_str)
        result = result.replace("${ABSTRACT}", abstract)
        result = result.replace("${FIRST_AUTHOR}", first_author)
        return result

    def render_section(self, title: str, content: str) -> str:
        """Render a paper section."""
        return f"\\section{{{title}}}\n{content}\n\n"

    def render_subsection(self, title: str, content: str) -> str:
        """Render a subsection."""
        return f"\\subsection{{{title}}}\n{content}\n\n"

    def render_closing(self) -> str:
        """Render document closing."""
        return r"""
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
"""

    def resolve_references(self, latex_doc: str) -> str:
        """Resolve cross-references (label → ref mapping)."""
        # Find all labels and create reference map
        labels = re.findall(r"\\label\{([^}]+)\}", latex_doc)
        # Ensure all refs exist
        for label in labels:
            if not re.search(rf"\\ref\{{{label}\}}", latex_doc):
                # Optionally warn about unused labels
                pass
        return latex_doc


class PaperSection(Enum):
    """Enum of standard academic paper sections."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"


@dataclass
class BibEntry:
    """Bibliography entry for citation management."""
    key: str
    entry_type: str  # article, inproceedings, misc, book, etc.
    title: str
    authors: list[str]
    year: int
    venue: str = ""
    url: str = ""
    doi: str = ""
    pages: str = ""
    volume: str = ""
    number: str = ""
    publisher: str = ""

    def to_bibtex(self) -> str:
        """Convert BibEntry to BibTeX format."""
        lines = [f"@{self.entry_type}{{{self.key},"]
        
        if self.title:
            lines.append(f'  title = {{{self.title}}},')
        
        if self.authors:
            authors_str = " and ".join(self.authors)
            lines.append(f'  author = {{{authors_str}}},')
        
        if self.year:
            lines.append(f'  year = {{{self.year}}},')
        
        if self.venue:
            venue_field = "journal" if self.entry_type == "article" else "booktitle"
            lines.append(f'  {venue_field} = {{{self.venue}}},')
        
        if self.volume:
            lines.append(f'  volume = {{{self.volume}}},')
        
        if self.number:
            lines.append(f'  number = {{{self.number}}},')
        
        if self.pages:
            lines.append(f'  pages = {{{self.pages}}},')
        
        if self.doi:
            lines.append(f'  doi = {{{self.doi}}},')
        
        if self.url:
            lines.append(f'  url = {{{self.url}}},')
        
        if self.publisher:
            lines.append(f'  publisher = {{{self.publisher}}},')
        
        lines[-1] = lines[-1].rstrip(",")  # Remove comma from last entry
        lines.append("}")
        
        return "\n".join(lines)


@dataclass
class FigureSpec:
    """Specification for a figure in the paper."""
    figure_id: str
    caption: str
    label: str
    file_path: Path
    width: str = "\\linewidth"
    position: str = "htbp"
    subfigures: list[FigureSpec] = field(default_factory=list)

    def to_latex(self) -> str:
        """Convert FigureSpec to LaTeX code."""
        if self.subfigures:
            return self._subfigures_to_latex()
        
        latex = f"\\begin{{figure}}[{self.position}]\n"
        latex += "  \\centering\n"
        latex += f"  \\includegraphics[width={self.width}]{{{self.file_path}}}\n"
        latex += f"  \\caption{{{self.caption}}}\n"
        latex += f"  \\label{{fig:{self.label}}}\n"
        latex += "\\end{figure}\n"
        
        return latex

    def _subfigures_to_latex(self) -> str:
        """Generate LaTeX for subfigures."""
        num_subfigs = len(self.subfigures)
        width_per_subfig = f"{0.95 / num_subfigs:.2f}\\linewidth"
        
        latex = f"\\begin{{figure}}[{self.position}]\n"
        latex += "  \\centering\n"
        
        for i, subfig in enumerate(self.subfigures):
            latex += f"  \\begin{{subfigure}}{{{width_per_subfig}}}\n"
            latex += "    \\centering\n"
            latex += f"    \\includegraphics[width=\\linewidth]{{{subfig.file_path}}}\n"
            latex += f"    \\caption{{{subfig.caption}}}\n"
            latex += f"    \\label{{fig:{subfig.label}}}\n"
            latex += "  \\end{subfigure}\n"
        
        latex += f"  \\caption{{{self.caption}}}\n"
        latex += f"  \\label{{fig:{self.label}}}\n"
        latex += "\\end{figure}\n"
        
        return latex


@dataclass
class TableSpec:
    """Specification for a table in the paper."""
    table_id: str
    caption: str
    label: str
    headers: list[str]
    rows: list[list[str]]
    position: str = "htbp"
    bold_best: bool = True

    def to_latex(self) -> str:
        """Convert TableSpec to LaTeX code."""
        # Determine number of columns
        num_cols = len(self.headers)
        
        # Build LaTeX table
        latex = f"\\begin{{table}}[{self.position}]\n"
        latex += "  \\centering\n"
        latex += f"  \\begin{{tabular}}{{{'|c' * num_cols}|}}\n"
        latex += "    \\hline\n"
        
        # Headers
        header_row = " & ".join(self.headers) + " \\\\\n"
        latex += f"    {header_row}"
        latex += "    \\hline\n"
        
        # Process rows with optional bolding of best values
        for row in self.rows:
            if self.bold_best and num_cols > 1:
                row = self._bold_best_values(row)
            row_str = " & ".join(row) + " \\\\\n"
            latex += f"    {row_str}"
        
        latex += "    \\hline\n"
        latex += "  \\end{tabular}\n"
        latex += f"  \\caption{{{self.caption}}}\n"
        latex += f"  \\label{{tab:{self.label}}}\n"
        latex += "\\end{table}\n"
        
        return latex

    def _bold_best_values(self, row: list[str]) -> list[str]:
        """Bold the best (maximum) values in each numeric column."""
        result = []
        
        for i, cell in enumerate(row):
            # Try to extract numeric value
            try:
                numeric_val = float(re.search(r"[\d.]+", cell).group())
                
                # Find max value in this column
                column_values = []
                for r in self.rows:
                    try:
                        val = float(re.search(r"[\d.]+", r[i]).group())
                        column_values.append(val)
                    except (ValueError, AttributeError):
                        pass
                
                if column_values and numeric_val == max(column_values):
                    result.append(f"\\textbf{{{cell}}}")
                else:
                    result.append(cell)
            except (ValueError, AttributeError):
                result.append(cell)
        
        return result


@dataclass
class SectionContent:
    """Content for a paper section."""
    section: PaperSection
    title: str
    content: str
    figures: list[FigureSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class PaperConfig:
    """Configuration for paper generation."""
    title: str
    authors: list[dict]  # Each dict: {name, affiliation, email}
    template: str = "neurips"  # neurips, icml, iclr, arxiv
    output_dir: Path = field(default_factory=lambda: Path("output/papers"))
    compile_pdf: bool = True
    max_pages: int = 10
    abstract_max_words: int = 250


class FigureGenerator:
    """Generate publication-quality figures from data or code."""

    async def generate_from_data(
        self,
        data: dict,
        figure_type: str,
        output_path: Path,
        llm: Any,
    ) -> FigureSpec:
        """Generate a figure from data using matplotlib code generation.
        
        Args:
            data: Dictionary with figure data and metadata
            figure_type: Type of figure (line_plot, bar_chart, scatter, heatmap, etc.)
            output_path: Where to save the figure
            llm: LLM instance for code generation
        
        Returns:
            FigureSpec describing the generated figure
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate matplotlib code
        code = self._generate_matplotlib_code(data, figure_type)
        
        # Execute code to generate figure
        await self.generate_from_code(code, output_path)
        
        # Create FigureSpec
        caption = data.get("caption", f"Generated {figure_type}")
        label = data.get("label", figure_type.lower())
        
        return FigureSpec(
            figure_id=label,
            caption=caption,
            label=label,
            file_path=output_path,
            width="0.85\\linewidth",
        )

    async def generate_from_code(self, code: str, output_path: Path) -> FigureSpec | None:
        """Execute matplotlib code to generate a figure.
        
        Args:
            code: Python code to execute
            output_path: Where to save the generated figure
        
        Returns:
            FigureSpec for the generated figure
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Execute code in subprocess
        full_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

{code}
plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
"""
        
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                logger.error(f"Figure generation failed: {result.stderr}")
                return None
            
            logger.info(f"Generated figure: {output_path}")
            return FigureSpec(
                figure_id="generated",
                caption="Generated figure",
                label="generated",
                file_path=output_path,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Figure generation timeout for {output_path}")
            return None

    def _generate_matplotlib_code(self, data: dict, figure_type: str) -> str:
        """Generate matplotlib code for different figure types.
        
        Supports: line_plot, bar_chart, scatter, heatmap, ablation_table, training_curve
        """
        if figure_type == "line_plot":
            return self._code_line_plot(data)
        elif figure_type == "bar_chart":
            return self._code_bar_chart(data)
        elif figure_type == "scatter":
            return self._code_scatter(data)
        elif figure_type == "heatmap":
            return self._code_heatmap(data)
        elif figure_type == "ablation_table":
            return self._code_ablation_table(data)
        elif figure_type == "training_curve":
            return self._code_training_curve(data)
        else:
            logger.warning(f"Unknown figure type: {figure_type}")
            return self._code_line_plot(data)

    @staticmethod
    def _code_line_plot(data: dict) -> str:
        """Generate code for line plot."""
        return """
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['x'], data['y'], marker='o', linewidth=2.5, markersize=8)
ax.set_xlabel(data.get('xlabel', 'X'), fontsize=12, fontweight='bold')
ax.set_ylabel(data.get('ylabel', 'Y'), fontsize=12, fontweight='bold')
ax.set_title(data.get('title', 'Line Plot'), fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
"""

    @staticmethod
    def _code_bar_chart(data: dict) -> str:
        """Generate code for bar chart."""
        return """
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(data['categories'], data['values'], color='steelblue', edgecolor='black', linewidth=1.5)
ax.set_xlabel(data.get('xlabel', 'Category'), fontsize=12, fontweight='bold')
ax.set_ylabel(data.get('ylabel', 'Value'), fontsize=12, fontweight='bold')
ax.set_title(data.get('title', 'Bar Chart'), fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
"""

    @staticmethod
    def _code_scatter(data: dict) -> str:
        """Generate code for scatter plot."""
        return """
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(data['x'], data['y'], c=data.get('colors', 'steelblue'), 
                     s=100, alpha=0.6, edgecolors='black', linewidth=1)
ax.set_xlabel(data.get('xlabel', 'X'), fontsize=12, fontweight='bold')
ax.set_ylabel(data.get('ylabel', 'Y'), fontsize=12, fontweight='bold')
ax.set_title(data.get('title', 'Scatter Plot'), fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
if 'colorbar' in data and data['colorbar']:
    plt.colorbar(scatter, ax=ax)
"""

    @staticmethod
    def _code_heatmap(data: dict) -> str:
        """Generate code for heatmap."""
        return """
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(data['matrix'], cmap='RdYlBu_r', aspect='auto')
ax.set_xlabel(data.get('xlabel', 'X'), fontsize=12, fontweight='bold')
ax.set_ylabel(data.get('ylabel', 'Y'), fontsize=12, fontweight='bold')
ax.set_title(data.get('title', 'Heatmap'), fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(data.get('colorbar_label', 'Value'), fontsize=11)
"""

    @staticmethod
    def _code_ablation_table(data: dict) -> str:
        """Generate code for ablation study visualization."""
        return """
fig, ax = plt.subplots(figsize=(12, 6))
components = data['components']
metrics = data['metrics']
x = np.arange(len(components))
width = 0.25
for i, metric in enumerate(metrics):
    values = [data[metric][comp] for comp in components]
    ax.bar(x + i*width, values, width, label=metric, edgecolor='black', linewidth=1)
ax.set_xlabel('Model Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(components, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
"""

    @staticmethod
    def _code_training_curve(data: dict) -> str:
        """Generate code for training curves."""
        return """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# Loss curve
ax1.plot(data['epochs'], data['train_loss'], 'o-', label='Train', linewidth=2.5)
if 'val_loss' in data:
    ax1.plot(data['epochs'], data['val_loss'], 's-', label='Validation', linewidth=2.5)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
# Accuracy curve
if 'accuracy' in data:
    ax2.plot(data['epochs'], data['accuracy'], 'o-', linewidth=2.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
"""


class BibManager:
    """Manage bibliography entries and generate BibTeX files."""

    def __init__(self) -> None:
        """Initialize bibliography manager."""
        self._entries: dict[str, BibEntry] = {}
        self.algorithm_ratio: float = 1.0  # Ratio of algorithmic vs LLM-fetched entries

    def add_entry(self, entry: BibEntry) -> None:
        """Add a bibliography entry.
        
        Args:
            entry: BibEntry to add
        """
        self._entries[entry.key] = entry
        logger.debug(f"Added bibliography entry: {entry.key}")

    async def add_from_semantic_scholar(self, paper_id: str) -> BibEntry | None:
        """Fetch metadata from Semantic Scholar and add entry.
        
        Args:
            paper_id: Semantic Scholar paper ID
        
        Returns:
            BibEntry created from metadata, or None on failure
        """
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                response = await client.get(
                    url,
                    params={
                        "fields": "paperId,title,authors,year,venue,citationCount,externalIds"
                    },
                    timeout=10,
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch from Semantic Scholar: {paper_id}")
                    return None
                
                data = response.json()
                
                # Parse authors
                authors = [author["name"] for author in data.get("authors", [])]
                
                # Create bibliography key
                first_author = authors[0].split()[-1].lower() if authors else "unknown"
                year = data.get("year", 0)
                key = f"{first_author}{year}"
                
                entry = BibEntry(
                    key=key,
                    entry_type="article",
                    title=data.get("title", ""),
                    authors=authors,
                    year=year,
                    venue=data.get("venue", ""),
                    url=data.get("externalIds", {}).get("ArXiv", ""),
                    doi=data.get("externalIds", {}).get("DOI", ""),
                )
                
                self.add_entry(entry)
                return entry
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {e}")
            return None

    def add_from_paper_reference(self, ref: Any) -> BibEntry:
        """Convert from literature_search.PaperReference to BibEntry.
        
        Args:
            ref: PaperReference object
        
        Returns:
            BibEntry created from the reference
        """
        authors = getattr(ref, "authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",")]
        
        key = self.get_citation_key(getattr(ref, "title", ""))
        
        entry = BibEntry(
            key=key,
            entry_type="article",
            title=getattr(ref, "title", ""),
            authors=authors,
            year=getattr(ref, "year", 0),
            venue=getattr(ref, "venue", ""),
            url=getattr(ref, "url", ""),
            doi=getattr(ref, "doi", ""),
        )
        
        self.add_entry(entry)
        return entry

    def generate_bib_file(self, output_path: Path) -> None:
        """Generate BibTeX file from entries.
        
        Args:
            output_path: Where to write the .bib file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for entry in self._entries.values():
            lines.append(entry.to_bibtex())
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated bibliography file: {output_path}")

    def get_citation_key(self, title: str) -> str:
        """Generate a citation key from a title.
        
        Args:
            title: Paper title
        
        Returns:
            Citation key (e.g., "smith2024")
        """
        words = title.lower().split()
        first_word = words[0] if words else "paper"
        key = first_word.replace(" ", "")
        year = 2024
        return f"{key}{year}"

    def get_entries(self) -> dict[str, BibEntry]:
        """Get all bibliography entries."""
        return self._entries.copy()

    def parse_bibtex_file(self, bib_path: Path) -> int:
        """Parse and merge entries from a .bib file.

        Args:
            bib_path: Path to .bib file

        Returns:
            Number of entries parsed
        """
        if not bib_path.exists():
            logger.warning(f"Bibliography file not found: {bib_path}")
            return 0

        try:
            content = bib_path.read_text(encoding="utf-8")
            # Simple regex-based BibTeX parsing
            entries = re.findall(
                r"@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\s*\}",
                content,
                re.DOTALL | re.IGNORECASE,
            )

            parsed_count = 0
            for entry_type, key, fields_str in entries:
                # Parse fields
                fields = {}
                for field_match in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", fields_str):
                    field_name = field_match.group(1).lower()
                    field_value = field_match.group(2).strip()
                    fields[field_name] = field_value

                # Extract authors
                authors = []
                if "author" in fields:
                    authors = [a.strip() for a in fields["author"].split(" and ")]

                entry = BibEntry(
                    key=key.strip(),
                    entry_type=entry_type.lower(),
                    title=fields.get("title", ""),
                    authors=authors,
                    year=int(fields.get("year", 0)) if fields.get("year") else 0,
                    venue=fields.get("journal", fields.get("booktitle", "")),
                    url=fields.get("url", ""),
                    doi=fields.get("doi", ""),
                    pages=fields.get("pages", ""),
                    volume=fields.get("volume", ""),
                    number=fields.get("number", ""),
                    publisher=fields.get("publisher", ""),
                )
                self.add_entry(entry)
                parsed_count += 1

            logger.info(f"Parsed {parsed_count} bibliography entries from {bib_path}")
            return parsed_count

        except Exception as e:
            logger.error(f"Error parsing bibliography file: {e}")
            return 0

    def merge_bibliography(self, other: "BibManager") -> None:
        """Merge entries from another BibManager, avoiding duplicates.

        Args:
            other: Another BibManager to merge from
        """
        for key, entry in other.get_entries().items():
            if key not in self._entries:
                self._entries[key] = entry


class SectionWriter:
    """Write individual paper sections with publication-quality content."""

    async def write_section(
        self,
        section: PaperSection,
        context: dict,
        llm: Any,
    ) -> SectionContent:
        """Write a section based on type.
        
        Args:
            section: PaperSection enum value
            context: Context dictionary with research details
            llm: LLM instance
        
        Returns:
            SectionContent with written section
        """
        if section == PaperSection.ABSTRACT:
            content = await self.write_abstract(context, llm)
            return SectionContent(
                section=section,
                title="Abstract",
                content=content,
            )
        elif section == PaperSection.INTRODUCTION:
            return await self.write_introduction(
                context,
                context.get("related_papers", []),
                llm,
            )
        elif section == PaperSection.RELATED_WORK:
            return await self.write_related_work(
                context.get("related_papers", []),
                context.get("our_contribution", ""),
                llm,
            )
        elif section == PaperSection.METHOD:
            return await self.write_method(
                context.get("method_description", ""),
                llm,
            )
        elif section == PaperSection.EXPERIMENTS:
            return await self.write_experiments(
                context.get("experiment_results", []),
                llm,
            )
        elif section == PaperSection.RESULTS:
            return await self.write_results(
                context.get("results_data", {}),
                llm,
            )
        elif section == PaperSection.DISCUSSION:
            return await self.write_discussion(
                context.get("findings", []),
                context.get("limitations", []),
                llm,
            )
        elif section == PaperSection.CONCLUSION:
            return await self.write_conclusion(
                context.get("findings", []),
                llm,
            )
        else:
            return SectionContent(
                section=section,
                title="Appendix",
                content="",
            )

    async def write_abstract(self, context: dict, llm: Any) -> str:
        """Write abstract section.
        
        Args:
            context: Research context
            llm: LLM instance
        
        Returns:
            Abstract text in LaTeX
        """
        prompt = f"""Write a concise, impactful abstract for an academic research paper.

Research Question: {context.get('research_question', '')}
Method: {context.get('method', '')}
Key Results: {context.get('key_results', '')}

Requirements:
- Maximum 250 words
- Formal academic tone, no first person singular
- One paragraph
- Include: problem, approach, results, implications
- Use \\cite{{key}} for citations
- No section headings or markdown

Abstract:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        return response.content

    async def write_introduction(
        self,
        context: dict,
        related_papers: list[BibEntry],
        llm: Any,
    ) -> SectionContent:
        """Write introduction section.
        
        Args:
            context: Research context
            related_papers: List of related BibEntry objects
            llm: LLM instance
        
        Returns:
            SectionContent for introduction
        """
        paper_refs = "\n".join([f"- {p.title} ({p.year})" for p in related_papers[:10]])
        
        prompt = f"""Write a compelling introduction for an academic research paper.

Problem: {context.get('problem_statement', '')}
Research Question: {context.get('research_question', '')}
Context: {context.get('background', '')}

Related Work:
{paper_refs}

Requirements:
- 400-600 words
- Formal academic tone
- Structure: problem motivation → context → research gap → contribution
- Use \\cite{{key}} for citations (use keys: {', '.join([p.key for p in related_papers[:5]])})
- No first person singular
- End with paper contributions and structure

Introduction:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        citations = [p.key for p in related_papers[:10]]
        
        return SectionContent(
            section=PaperSection.INTRODUCTION,
            title="Introduction",
            content=response.content,
            citations=citations,
        )

    async def write_related_work(
        self,
        papers: list[BibEntry],
        our_contribution: str,
        llm: Any,
    ) -> SectionContent:
        """Write related work section.
        
        Args:
            papers: List of related papers
            our_contribution: Description of our contribution
            llm: LLM instance
        
        Returns:
            SectionContent for related work
        """
        paper_summaries = "\n".join(
            [f"- {p.title}: {p.venue or 'Unknown'} ({p.year})" for p in papers[:15]]
        )
        
        prompt = f"""Write a comprehensive related work section that positions this research.

Our Contribution: {our_contribution}

Related Papers:
{paper_summaries}

Requirements:
- 500-800 words
- Group related work by themes/approaches
- Formal academic tone
- Use \\cite{{key}} for citations (available keys: {', '.join([p.key for p in papers[:10]])})
- Highlight novelty compared to prior work
- No first person singular
- End with what distinguishes our work

Related Work:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        citations = [p.key for p in papers]
        
        return SectionContent(
            section=PaperSection.RELATED_WORK,
            title="Related Work",
            content=response.content,
            citations=citations,
        )

    async def write_method(self, method_description: str, llm: Any) -> SectionContent:
        """Write method section.
        
        Args:
            method_description: Description of the method
            llm: LLM instance
        
        Returns:
            SectionContent for methods
        """
        prompt = f"""Write a clear, detailed methods section for an academic paper.

Method Overview: {method_description}

Requirements:
- 600-1000 words
- Divide into logical subsections with \\subsection{{}} if needed
- Technical but accessible
- Include: algorithm/approach, key components, design choices, implementation details
- Use mathematical notation where appropriate ($$...$$)
- Formal academic tone, no first person singular
- Reference figures if needed: \\ref{{fig:label}}

Method Section:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        return SectionContent(
            section=PaperSection.METHOD,
            title="Method",
            content=response.content,
        )

    async def write_experiments(
        self,
        experiment_results: list[dict],
        llm: Any,
    ) -> SectionContent:
        """Write experiments section.
        
        Args:
            experiment_results: List of experiment result dictionaries
            llm: LLM instance
        
        Returns:
            SectionContent for experiments
        """
        results_summary = "\n".join(
            [f"- {exp.get('name', 'Exp')}: {exp.get('metric', 'N/A')}" 
             for exp in experiment_results[:10]]
        )
        
        prompt = f"""Write an experiments section describing research methodology and setup.

Experiments:
{results_summary}

Requirements:
- 400-600 words
- Cover: datasets, baselines, metrics, experimental setup, protocols
- Subsections: \\subsection{{}} for different experiments
- Formal tone, no first person singular
- Reference tables/figures with \\ref{{tab:label}} or \\ref{{fig:label}}
- Explain why these experiments are important
- No actual results (save for Results section)

Experiments Section:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        return SectionContent(
            section=PaperSection.EXPERIMENTS,
            title="Experiments",
            content=response.content,
        )

    async def write_results(self, results_data: dict, llm: Any) -> SectionContent:
        """Write results section.
        
        Args:
            results_data: Dictionary with results
            llm: LLM instance
        
        Returns:
            SectionContent for results
        """
        results_text = str(results_data)
        
        prompt = f"""Write a results section presenting experimental findings.

Results Summary: {results_text}

Requirements:
- 400-700 words
- Present: main results, comparisons, performance metrics
- Reference tables (\\ref{{tab:label}}) and figures (\\ref{{fig:label}})
- Objective, factual tone
- No interpretation (save for Discussion)
- Use quantitative language
- Formal academic style
- Structure by result type or experiment

Results Section:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        return SectionContent(
            section=PaperSection.RESULTS,
            title="Results",
            content=response.content,
        )

    async def write_discussion(
        self,
        findings: list[str],
        limitations: list[str],
        llm: Any,
    ) -> SectionContent:
        """Write discussion section.
        
        Args:
            findings: Key findings
            limitations: Limitations of the work
            llm: LLM instance
        
        Returns:
            SectionContent for discussion
        """
        findings_text = "\n".join([f"- {f}" for f in findings[:10]])
        limitations_text = "\n".join([f"- {l}" for l in limitations[:5]])
        
        prompt = f"""Write a discussion section interpreting results and their significance.

Key Findings:
{findings_text}

Limitations:
{limitations_text}

Requirements:
- 500-800 words
- Interpret results in context of research questions
- Compare with related work
- Discuss implications
- Address limitations honestly
- Discuss future work
- Formal academic tone
- Use technical language appropriately
- Reference results (\\ref{{tab:label}}, \\ref{{fig:label}})

Discussion Section:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        return SectionContent(
            section=PaperSection.DISCUSSION,
            title="Discussion",
            content=response.content,
        )

    async def write_conclusion(
        self,
        findings: list[str],
        llm: Any,
    ) -> SectionContent:
        """Write conclusion section.
        
        Args:
            findings: Key findings to summarize
            llm: LLM instance
        
        Returns:
            SectionContent for conclusion
        """
        findings_text = "\n".join([f"- {f}" for f in findings[:5]])
        
        prompt = f"""Write a concise conclusion summarizing the paper's contributions.

Key Findings:
{findings_text}

Requirements:
- 200-300 words
- Summarize research question and approach
- Highlight key contributions
- Discuss broader impact
- Suggest future directions
- Formal academic tone
- No new information
- Strong closing statement

Conclusion:"""
        
        response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
        
        return SectionContent(
            section=PaperSection.CONCLUSION,
            title="Conclusion",
            content=response.content,
        )


class LatexCompiler:
    """Compile LaTeX documents to PDF."""

    async def compile(self, tex_path: Path, output_dir: Path) -> Path | None:
        """Compile LaTeX document to PDF.
        
        Args:
            tex_path: Path to .tex file
            output_dir: Directory for output PDF
        
        Returns:
            Path to compiled PDF or None on failure
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Change to output directory for compilation
            original_cwd = Path.cwd()
            
            # Run pdflatex multiple times for references
            commands = [
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                ["bibtex", str(output_dir / tex_path.stem)],
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            ]
            
            for cmd in commands:
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=output_dir,
                )
                
                if result.returncode != 0:
                    logger.warning(f"Compilation step failed: {' '.join(cmd)}")
                    # Continue trying - not all steps are critical
            
            pdf_path = output_dir / f"{tex_path.stem}.pdf"
            
            if pdf_path.exists():
                logger.info(f"Successfully compiled PDF: {pdf_path}")
                return pdf_path
            else:
                logger.error(f"PDF compilation did not produce output file: {pdf_path}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"LaTeX compilation timeout for {tex_path}")
            return None
        except Exception as e:
            logger.error(f"LaTeX compilation error: {e}")
            return None


class PaperWriter:
    """Main orchestrator for academic paper generation."""

    NEURIPS_PREAMBLE = r"""
\documentclass{article}
\usepackage[utf-8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{float}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[margin=1in]{geometry}

\title{%TITLE%}
\author{%AUTHORS%}
\date{\today}

\begin{document}

\maketitle
"""

    ICML_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{float}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[margin=1in]{geometry}

\title{%TITLE%}
\author{%AUTHORS%}
\date{}

\begin{document}

\maketitle
"""

    ICLR_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{float}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}

\title{%TITLE%}
\author{%AUTHORS%}
\date{}

\begin{document}

\maketitle
"""

    ARXIV_PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[utf-8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{float}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}

\title{%TITLE%}
\author{%AUTHORS%}
\date{\today}

\begin{document}

\maketitle
"""

    def __init__(self, config: PaperConfig) -> None:
        """Initialize paper writer.
        
        Args:
            config: PaperConfig instance
        """
        self.config = config
        self.bib_manager = BibManager()
        self.section_writer = SectionWriter()
        self.figure_generator = FigureGenerator()
        self.latex_compiler = LatexCompiler()
        self.sections: list[SectionContent] = []

    async def write_paper(self, context: dict, llm: Any) -> Path | None:
        """Generate a complete paper from research context.
        
        Args:
            context: Research context with:
                - research_question: str
                - method: str
                - experiment_results: list[dict]
                - figures_data: list[dict]
                - related_papers: list[BibEntry]
                - findings: list[str]
                - limitations: list[str]
            llm: LLM instance
        
        Returns:
            Path to generated PDF or None on failure
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate all sections
            logger.info("Generating paper sections...")
            self.sections = await self._generate_all_sections(context, llm)
            
            # Add bibliography entries
            logger.info("Building bibliography...")
            for paper in context.get("related_papers", []):
                self.bib_manager.add_entry(paper)
            
            # Assemble LaTeX document
            logger.info("Assembling LaTeX document...")
            bib_path = self.config.output_dir / "references.bib"
            self.bib_manager.generate_bib_file(bib_path)
            
            latex_content = self._assemble_latex(self.sections, bib_path)
            
            # Write .tex file
            tex_path = self.config.output_dir / "paper.tex"
            tex_path.write_text(latex_content, encoding="utf-8")
            logger.info(f"Wrote LaTeX document: {tex_path}")
            
            # Compile to PDF
            if self.config.compile_pdf:
                logger.info("Compiling LaTeX to PDF...")
                pdf_path = await self.latex_compiler.compile(tex_path, self.config.output_dir)
                return pdf_path
            else:
                return tex_path
                
        except Exception as e:
            logger.error(f"Paper generation failed: {e}")
            return None

    async def _generate_all_sections(
        self,
        context: dict,
        llm: Any,
    ) -> list[SectionContent]:
        """Generate all paper sections.
        
        Args:
            context: Research context
            llm: LLM instance
        
        Returns:
            List of SectionContent
        """
        sections = []
        
        # Generate main sections in order
        section_order = [
            PaperSection.ABSTRACT,
            PaperSection.INTRODUCTION,
            PaperSection.RELATED_WORK,
            PaperSection.METHOD,
            PaperSection.EXPERIMENTS,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION,
        ]
        
        for section in section_order:
            try:
                logger.info(f"Writing {section.value} section...")
                content = await self.section_writer.write_section(section, context, llm)
                sections.append(content)
            except Exception as e:
                logger.error(f"Failed to write {section.value}: {e}")
                # Continue with other sections
        
        return sections

    def _assemble_latex(self, sections: list[SectionContent], bib_path: Path) -> str:
        """Assemble LaTeX document from sections.
        
        Args:
            sections: List of SectionContent
            bib_path: Path to .bib file
        
        Returns:
            Complete LaTeX document
        """
        # Get preamble
        preamble = self._get_template_preamble()
        
        # Format authors
        authors_str = " \\\\ ".join(
            [f"{a['name']} ({a['affiliation']})" for a in self.config.authors]
        )
        preamble = preamble.replace("%TITLE%", self.config.title)
        preamble = preamble.replace("%AUTHORS%", authors_str)
        
        # Build document body
        body_lines = [preamble]
        
        for section in sections:
            if section.section == PaperSection.ABSTRACT:
                body_lines.append("\\begin{abstract}")
                body_lines.append(section.content)
                body_lines.append("\\end{abstract}\n")
            else:
                # Add section heading
                if section.section == PaperSection.APPENDIX:
                    body_lines.append("\\appendix")
                
                body_lines.append(f"\\section{{{section.title}}}")
                body_lines.append(section.content)
                body_lines.append("")
                
                # Add tables and figures
                for table in section.tables:
                    body_lines.append(table.to_latex())
                
                for figure in section.figures:
                    body_lines.append(figure.to_latex())
        
        # Add bibliography
        body_lines.append("\\newpage")
        body_lines.append("\\bibliographystyle{plainnat}")
        body_lines.append(f"\\bibliography{{{bib_path.stem}}}")
        
        # End document
        body_lines.append("\\end{document}")
        
        return "\n".join(body_lines)

    def _get_template_preamble(self) -> str:
        """Get LaTeX preamble for selected template.
        
        Returns:
            LaTeX preamble string
        """
        template = self.config.template.lower()
        
        if template == "neurips":
            return self.NEURIPS_PREAMBLE
        elif template == "icml":
            return self.ICML_PREAMBLE
        elif template == "iclr":
            return self.ICLR_PREAMBLE
        elif template == "arxiv":
            return self.ARXIV_PREAMBLE
        else:
            logger.warning(f"Unknown template: {template}, using NeurIPS")
            return self.NEURIPS_PREAMBLE

    def get_paper_stats(self) -> dict:
        """Get statistics about the generated paper.
        
        Returns:
            Dictionary with word_count, figure_count, table_count, citation_count
        """
        word_count = sum(len(s.content.split()) for s in self.sections)
        figure_count = sum(len(s.figures) for s in self.sections)
        table_count = sum(len(s.tables) for s in self.sections)
        citation_count = len(self.bib_manager.get_entries())
        
        return {
            "word_count": word_count,
            "figure_count": figure_count,
            "table_count": table_count,
            "citation_count": citation_count,
            "section_count": len(self.sections),
        }
