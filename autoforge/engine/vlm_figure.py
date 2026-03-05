"""Vision-Language Model (VLM) Figure Analysis Pipeline.

This module implements a comprehensive VLM-powered figure analysis system for
automated extraction, analysis, comparison, and reproduction of figures from
academic papers and technical documents.

It integrates with Claude's vision capabilities to:
- Extract and classify figures from PDFs
- Analyze figure content and extract data points
- Verify figure claims against article text
- Reproduce figures from their analysis
- Cross-reference numerical data

Design philosophy:
  1. Multi-modal analysis: Combine OCR, image processing, and VLM reasoning
  2. Extraction-first: Get images and metadata from PDFs robustly
  3. Iterative refinement: Use feedback loops to fix OCR errors and improve analyses
  4. Reproducibility: Generate code to recreate figures for validation
  5. Verification: Check figure claims against stated results in text

Architecture:
  - FigureExtractor: PDF → figures + metadata
  - VLMAnalyzer: image + caption → rich analysis
  - FigureReproducer: analysis → executable code → figure image
  - FigureVerifier: analysis + claims → verification report
  - VLMFigurePipeline: End-to-end orchestration
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Optional imports for PDF and image processing
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Enumerations & Data Structures
# ══════════════════════════════════════════════════════════════


class FigureType(str, Enum):
    """Classification of figure types."""
    PLOT = "plot"
    DIAGRAM = "diagram"
    TABLE = "table"
    EQUATION = "equation"
    ARCHITECTURE = "architecture"
    FLOWCHART = "flowchart"
    PHOTOGRAPH = "photograph"
    SCHEMATIC = "schematic"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BAR_CHART = "bar_chart"
    OTHER = "other"


@dataclass
class FigureMetadata:
    """Metadata for a figure extracted from a document."""
    figure_id: str
    figure_type: FigureType
    caption: str
    label: str
    page_number: int = 0
    bounding_box: tuple[float, float, float, float] | None = None
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type.value,
            "caption": self.caption,
            "label": self.label,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box,
            "source_path": str(self.source_path) if self.source_path else None,
        }


@dataclass
class FigureAnalysis:
    """Comprehensive analysis of a figure produced by VLM."""
    metadata: FigureMetadata
    description: str
    data_points: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    mathematical_content: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_vlm_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "description": self.description,
            "data_points": self.data_points,
            "relationships": self.relationships,
            "key_findings": self.key_findings,
            "mathematical_content": self.mathematical_content,
            "anomalies": self.anomalies,
            "confidence": self.confidence,
        }


@dataclass
class VerificationResult:
    """Result of figure claim verification."""
    claim: str
    supported: bool
    confidence: float
    evidence: str
    contradictions: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# Figure Extraction
# ══════════════════════════════════════════════════════════════


class FigureExtractor:
    """Extract figures and captions from PDF documents."""

    def __init__(self, prefer_pdfplumber: bool = True):
        """Initialize extractor with backend preference."""
        self.prefer_pdfplumber = prefer_pdfplumber
        self.logger = logging.getLogger(f"{__name__}.FigureExtractor")

    async def extract_figures(
        self,
        pdf_path: Path,
    ) -> list[FigureMetadata]:
        """Extract figures from PDF using available backends.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of FigureMetadata with extracted figure information
        """
        if not pdf_path.exists():
            self.logger.error(f"PDF not found: {pdf_path}")
            return []

        figures = []

        # Try preferred backend first
        if self.prefer_pdfplumber and pdfplumber is not None:
            figures = await asyncio.to_thread(
                self._extract_with_pdfplumber, pdf_path
            )
        elif fitz is not None:
            figures = await asyncio.to_thread(
                self._extract_with_pymupdf, pdf_path
            )
        else:
            self.logger.warning(
                "No PDF backend available (install pdfplumber or pymupdf)"
            )
            return []

        # Enrich with captions
        captions = await self._extract_captions_from_pdf(pdf_path)
        figures = self._map_captions_to_figures(figures, captions)

        return figures

    def _extract_with_pdfplumber(self, pdf_path: Path) -> list[FigureMetadata]:
        """Extract figures using pdfplumber backend."""
        figures = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Look for images in the page
                if hasattr(page, "images"):
                    for img_idx, img in enumerate(page.images):
                        figure_id = f"fig_{page_num}_{img_idx}"
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])

                        metadata = FigureMetadata(
                            figure_id=figure_id,
                            figure_type=FigureType.OTHER,
                            caption="",
                            label=figure_id,
                            page_number=page_num,
                            bounding_box=bbox,
                            source_path=pdf_path,
                        )
                        figures.append(metadata)

                # Try to detect figure regions from layout
                detected_regions = self._detect_figure_regions(page)
                for region_idx, region in enumerate(detected_regions):
                    figure_id = f"fig_region_{page_num}_{region_idx}"
                    metadata = FigureMetadata(
                        figure_id=figure_id,
                        figure_type=FigureType.OTHER,
                        caption="",
                        label=figure_id,
                        page_number=page_num,
                        bounding_box=region,
                        source_path=pdf_path,
                    )
                    figures.append(metadata)

        return figures

    def _extract_with_pymupdf(self, pdf_path: Path) -> list[FigureMetadata]:
        """Extract figures using PyMuPDF backend."""
        figures = []

        doc = fitz.open(pdf_path)
        try:
            for page_num, page in enumerate(doc, start=1):
                # Extract images from page
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    figure_id = f"fig_{page_num}_{img_idx}"
                    # get_image_bbox expects the full image tuple, not just xref
                    bbox = page.get_image_bbox(img)

                    metadata = FigureMetadata(
                        figure_id=figure_id,
                        figure_type=FigureType.OTHER,
                        caption="",
                        label=figure_id,
                        page_number=page_num,
                        bounding_box=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        source_path=pdf_path,
                    )
                    figures.append(metadata)

                # Detect additional figure regions
                detected_regions = self._detect_figure_regions(page)
                for region_idx, region in enumerate(detected_regions):
                    figure_id = f"fig_region_{page_num}_{region_idx}"
                    metadata = FigureMetadata(
                        figure_id=figure_id,
                        figure_type=FigureType.OTHER,
                        caption="",
                        label=figure_id,
                        page_number=page_num,
                        bounding_box=region,
                        source_path=pdf_path,
                    )
                    figures.append(metadata)
        finally:
            doc.close()

        return figures

    def _detect_figure_regions(
        self,
        page: Any,
    ) -> list[tuple[float, float, float, float]]:
        """Detect figure regions on a page using heuristics.

        Simple heuristic: look for large non-text regions that are likely
        to contain figures, plots, diagrams, etc.
        """
        width = getattr(page, "width", None)
        height = getattr(page, "height", None)
        if width is None and hasattr(page, "rect"):
            rect = page.rect
            width = getattr(rect, "width", None)
            height = getattr(rect, "height", None)

        if not width or not height:
            return []

        page_area = float(width) * float(height)
        regions: list[tuple[float, float, float, float]] = []
        min_area = page_area * 0.012
        min_size = 40.0
        max_regions = 18

        def _clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        def _iof(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            if ax1 <= ax0 or ay1 <= ay0:
                return 0.0
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0)
            area_a = (ax1 - ax0) * (ay1 - ay0)
            area_b = (bx1 - bx0) * (by1 - by0)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def _add_if_region(region: tuple[float, float, float, float]) -> None:
            x0, y0, x1, y1 = (
                _clamp(float(region[0]), 0.0, float(width)),
                _clamp(float(region[1]), 0.0, float(height)),
                _clamp(float(region[2]), 0.0, float(width)),
                _clamp(float(region[3]), 0.0, float(height)),
            )

            if not (x1 > x0 and y1 > y0):
                return
            w = x1 - x0
            h = y1 - y0
            if w < min_size or h < min_size:
                return
            if w * h < min_area:
                return

            normalized = (x0, y0, x1, y1)
            for existing in regions:
                if _iof(existing, normalized) >= 0.45:
                    return

            regions.append(normalized)

        # PDF image objects (most reliable signal)
        if hasattr(page, "images") and page.images:
            for image in page.images:
                if not isinstance(image, dict):
                    continue
                x0 = image.get("x0")
                y0 = image.get("y0", image.get("top"))
                x1 = image.get("x1")
                y1 = image.get("y1", image.get("bottom"))
                if None in (x0, y0, x1, y1):
                    continue
                _add_if_region((x0, y0, x1, y1))

        # Vector geometry rectangles from PDF pages (often chart-like containers)
        if hasattr(page, "rects"):
            for rect in page.rects:
                if not isinstance(rect, dict):
                    continue
                x0 = rect.get("x0")
                y0 = rect.get("y0", rect.get("top"))
                x1 = rect.get("x1")
                y1 = rect.get("y1", rect.get("bottom"))
                if None in (x0, y0, x1, y1):
                    continue
                _add_if_region((x0, y0, x1, y1))

        # FitZ text-dict image blocks if available
        if hasattr(page, "get_text"):
            try:
                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type") != 1:
                        continue
                    block_bbox = block.get("bbox")
                    if not block_bbox or len(block_bbox) != 4:
                        continue
                    _add_if_region(block_bbox)
            except Exception:
                pass

        if regions:
            return regions[:max_regions]

        return []

    async def _extract_captions_from_pdf(
        self,
        pdf_path: Path,
    ) -> dict[int, list[str]]:
        """Extract figure captions from PDF text.

        Returns:
            Mapping of page number → list of captions on that page
        """
        if pdfplumber is None:
            return {}

        return await asyncio.to_thread(self._extract_captions_sync, pdf_path)

    def _extract_captions_sync(self, pdf_path: Path) -> dict[int, list[str]]:
        """Synchronous caption extraction (run via to_thread)."""
        captions: dict[int, list[str]] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Match "Figure X:" or "Fig. X:" including multiline captions
                pattern = r"(?:Figure|Fig\.?)\s+(\d+)[:\.]?\s*(.+?)(?=\n\s*\n|(?:Figure|Fig\.)\s+\d|$)"
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    captions[page_num] = [
                        (num, caption.strip()) for num, caption in matches
                    ]

        return captions

    def _map_captions_to_figures(
        self,
        figures: list[FigureMetadata],
        captions: dict[int, list[str]],
    ) -> list[FigureMetadata]:
        """Map extracted captions to figures by page.

        Assigns captions to figures in order within each page.
        """
        # Group figures by page
        page_figures: dict[int, list[FigureMetadata]] = {}
        for figure in figures:
            page_figures.setdefault(figure.page_number, []).append(figure)

        for page_num, page_figs in page_figures.items():
            page_captions = captions.get(page_num, [])
            for idx, fig in enumerate(page_figs):
                if idx < len(page_captions):
                    _num, caption_text = page_captions[idx]
                    fig.caption = caption_text

        return figures


# ══════════════════════════════════════════════════════════════
# VLM Analysis
# ══════════════════════════════════════════════════════════════


class VLMAnalyzer:
    """Analyze figures using Vision-Language Models."""

    def __init__(self, llm: Any):
        """Initialize analyzer with LLM client."""
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.VLMAnalyzer")

    async def analyze_figure(
        self,
        image_path: Path,
        caption: str = "",
        context: str = "",
    ) -> FigureAnalysis:
        """Analyze a single figure using VLM.

        Args:
            image_path: Path to figure image
            caption: Figure caption (optional)
            context: Additional context about the paper/section

        Returns:
            FigureAnalysis with detailed extraction and insights
        """
        if not image_path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return FigureAnalysis(
                metadata=FigureMetadata(
                    figure_id="unknown",
                    figure_type=FigureType.OTHER,
                    caption=caption,
                    label="unknown",
                ),
                description="",
            )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(caption, context)

        # Call VLM with image
        try:
            response = await self._call_vlm_with_image(image_path, prompt)

            analysis_dict = self._parse_analysis_response(response)

            # Create metadata — gracefully handle unknown figure types from VLM
            try:
                fig_type = FigureType(analysis_dict.get("figure_type", "other"))
            except ValueError:
                fig_type = FigureType.OTHER

            metadata = FigureMetadata(
                figure_id=image_path.stem,
                figure_type=fig_type,
                caption=caption,
                label=image_path.stem,
                source_path=image_path,
            )

            return FigureAnalysis(
                metadata=metadata,
                description=analysis_dict.get("description", ""),
                data_points=analysis_dict.get("data_points", []),
                relationships=analysis_dict.get("relationships", []),
                key_findings=analysis_dict.get("key_findings", []),
                mathematical_content=analysis_dict.get("mathematical_content", []),
                anomalies=analysis_dict.get("anomalies", []),
                confidence=analysis_dict.get("confidence", 0.5),
                raw_vlm_response=response,
            )
        except Exception as e:
            self.logger.error(f"Error analyzing figure {image_path}: {e}")
            return FigureAnalysis(
                metadata=FigureMetadata(
                    figure_id=image_path.stem,
                    figure_type=FigureType.OTHER,
                    caption=caption,
                    label=image_path.stem,
                    source_path=image_path,
                ),
                description="",
            )

    async def analyze_batch(
        self,
        figures: list[tuple[Path, str]],
    ) -> list[FigureAnalysis]:
        """Analyze multiple figures in parallel.

        Args:
            figures: List of (image_path, caption) tuples

        Returns:
            List of FigureAnalysis results
        """
        tasks = [
            self.analyze_figure(img_path, caption)
            for img_path, caption in figures
        ]
        return await asyncio.gather(*tasks)

    async def compare_figures(
        self,
        fig1: FigureAnalysis,
        fig2: FigureAnalysis,
    ) -> dict[str, Any]:
        """Compare two figures for similarities and differences.

        Args:
            fig1, fig2: FigureAnalysis objects to compare

        Returns:
            Comparison report with similarities and differences
        """
        prompt = f"""Compare these two figures:

Figure 1:
Type: {fig1.metadata.figure_type.value}
Caption: {fig1.metadata.caption}
Description: {fig1.description}
Key Findings: {fig1.key_findings}

Figure 2:
Type: {fig2.metadata.figure_type.value}
Caption: {fig2.metadata.caption}
Description: {fig2.description}
Key Findings: {fig2.key_findings}

Provide:
1. Similarities in data/content
2. Differences in methodology/representation
3. Relationship between findings
4. Consistency assessment

Return as JSON with keys: similarities, differences, relationship, consistency."""

        try:
            response = await self._call_llm_text(prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error comparing figures: {e}")
            return {
                "similarities": [],
                "differences": [],
                "relationship": "",
                "consistency": "unknown",
            }

    async def extract_data_from_plot(
        self,
        image_path: Path,
    ) -> list[dict[str, Any]]:
        """Extract numerical data points from a plot/chart.

        Args:
            image_path: Path to plot image

        Returns:
            List of extracted data points with x, y, labels, etc.
        """
        prompt = """Extract all numerical data from this plot. For each data point or series:
- If it's an x-y plot: extract (x, y) pairs
- If it's a table: extract all cell values
- If it's a bar chart: extract bar heights and labels
- Include error bars, confidence intervals if present

Return as JSON array with objects like:
{
  "x": value or label,
  "y": value,
  "error": optional,
  "series": "name if multiple series",
  "unit": "if visible"
}

Only return valid JSON."""

        try:
            response = await self._call_vlm_with_image(image_path, prompt)

            # Try to extract JSON from response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except Exception as e:
            self.logger.error(f"Error extracting data from plot: {e}")
            return []

    def _build_analysis_prompt(self, caption: str, context: str) -> str:
        """Build VLM analysis prompt."""
        prompt = """Analyze this figure carefully. Provide detailed analysis in the following JSON format:

{
  "figure_type": "one of: plot, diagram, table, equation, architecture, flowchart, photograph, schematic, heatmap, scatter, bar_chart, other",
  "description": "Detailed natural language description of what the figure shows",
  "data_points": [{"label": "...", "value": ...}, ...],
  "relationships": ["Variable X correlates with Y", ...],
  "key_findings": ["Main insight from figure", ...],
  "mathematical_content": ["Any equations or formulas visible", ...],
  "anomalies": ["Any unusual patterns or potential issues", ...],
  "confidence": 0.0-1.0
}"""

        if caption:
            prompt += f"\n\nCaption: {caption}"
        if context:
            prompt += f"\n\nContext: {context}"

        return prompt

    def _parse_analysis_response(self, response: str) -> dict[str, Any]:
        """Parse VLM analysis response."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse analysis response: {e}")

        # Fallback: return minimal structure
        return {
            "figure_type": "other",
            "description": response,
            "confidence": 0.5,
        }

    async def _call_vlm_with_image(self, image_path: Path, prompt: str) -> str:
        """Call VLM with an image via the LLM router.

        Reads the image file, base64-encodes it, and sends it as an image
        content block alongside the text prompt.  Works with any provider
        that supports vision (Anthropic Claude, OpenAI GPT-4V, Gemini).
        """
        import base64
        import mimetypes

        if self.llm is None:
            raise RuntimeError(
                "VLMAnalyzer requires an LLM router instance (self.llm). "
                "Pass one to the constructor."
            )

        # Read and encode the image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_bytes = image_path.read_bytes()
        b64_data = base64.b64encode(raw_bytes).decode("ascii")
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"

        # Build a user message with image + text content blocks
        from autoforge.engine.llm_router import ContentBlock, TaskComplexity

        messages = [
            {
                "role": "user",
                "content": [
                    ContentBlock(type="image", media_type=mime, image_data=b64_data),
                    ContentBlock(type="text", text=prompt),
                ],
            }
        ]

        resp = await self.llm.call(
            complexity=TaskComplexity.HIGH,
            system="You are an expert academic figure analyst. Respond in valid JSON.",
            messages=messages,
        )
        # Extract text from response content blocks
        parts = []
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "\n".join(parts)

    async def _call_llm_text(self, prompt: str) -> str:
        """Call LLM with a text-only prompt via the LLM router."""
        if self.llm is None:
            raise RuntimeError(
                "VLMAnalyzer requires an LLM router instance (self.llm). "
                "Pass one to the constructor."
            )

        from autoforge.engine.llm_router import TaskComplexity

        messages = [{"role": "user", "content": prompt}]

        resp = await self.llm.call(
            complexity=TaskComplexity.STANDARD,
            system="You are an expert academic figure analyst. Respond in valid JSON.",
            messages=messages,
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# Figure Reproduction
# ══════════════════════════════════════════════════════════════


class FigureReproducer:
    """Generate code to reproduce figures from their analysis."""

    def __init__(self, llm: Any):
        """Initialize reproducer with LLM client."""
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.FigureReproducer")

    async def reproduce_figure(
        self,
        analysis: FigureAnalysis,
        output_dir: Path,
    ) -> Path | None:
        """Generate and execute code to reproduce a figure.

        Args:
            analysis: FigureAnalysis containing data and metadata
            output_dir: Directory to save reproduced figure

        Returns:
            Path to generated figure image, or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate code based on figure type
        code = await self._generate_plot_code(analysis)
        if not code:
            self.logger.warning(
                f"Failed to generate code for {analysis.metadata.figure_id}"
            )
            return None

        # Try to execute the code
        output_path = output_dir / f"{analysis.metadata.figure_id}.png"
        success = await self._execute_plot_code(code, output_path)

        if not success:
            self.logger.warning(
                f"Failed to execute code for {analysis.metadata.figure_id}"
            )
            return None

        return output_path

    async def _generate_plot_code(
        self,
        analysis: FigureAnalysis,
    ) -> str:
        """Generate Python code to reproduce the figure."""
        if analysis.metadata.figure_type == FigureType.TABLE:
            return await self._generate_table_code(analysis)
        elif analysis.metadata.figure_type == FigureType.PLOT:
            return await self._generate_line_plot_code(analysis)
        elif analysis.metadata.figure_type in (
            FigureType.BAR_CHART,
            FigureType.SCATTER,
        ):
            return await self._generate_chart_code(analysis)
        else:
            return ""

    async def _generate_line_plot_code(
        self,
        analysis: FigureAnalysis,
    ) -> str:
        """Generate code for line plot reproduction."""
        if not analysis.data_points:
            return ""

        # Extract x and y values
        x_values = [p.get("x", 0) for p in analysis.data_points]
        y_values = [p.get("y", 0) for p in analysis.data_points]

        safe_caption = repr(analysis.metadata.caption)
        code = f'''import matplotlib.pyplot as plt
import numpy as np

x = {x_values!r}
y = {y_values!r}

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-o', linewidth=2, markersize=6)
plt.title({safe_caption})
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("{{output_path}}", dpi=150, bbox_inches="tight")
plt.close()
'''
        return code

    async def _generate_table_code(
        self,
        analysis: FigureAnalysis,
    ) -> str:
        """Generate code for table reproduction."""
        if not analysis.data_points:
            return ""

        safe_caption = repr(analysis.metadata.caption)
        code = f'''import matplotlib.pyplot as plt

data = {analysis.data_points!r}

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table_data = [[str(item.get(k, '')) for k in item.keys()] for item in data]
table = ax.table(cellText=table_data, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.title({safe_caption})
plt.tight_layout()
plt.savefig("{{output_path}}", dpi=150, bbox_inches="tight")
plt.close()
'''
        return code

    async def _generate_chart_code(
        self,
        analysis: FigureAnalysis,
    ) -> str:
        """Generate code for bar/scatter chart reproduction."""
        if not analysis.data_points:
            return ""

        safe_caption = repr(analysis.metadata.caption)
        is_bar = analysis.metadata.figure_type == FigureType.BAR_CHART
        code = f'''import matplotlib.pyplot as plt

data = {analysis.data_points!r}

plt.figure(figsize=(10, 6))
if {is_bar!r}:
    labels = [str(p.get("x", "")) for p in data]
    values = [p.get("y", 0) for p in data]
    plt.bar(labels, values)
else:  # scatter
    x = [p.get("x", 0) for p in data]
    y = [p.get("y", 0) for p in data]
    plt.scatter(x, y)

plt.title({safe_caption})
plt.tight_layout()
plt.savefig("{{output_path}}", dpi=150, bbox_inches="tight")
plt.close()
'''
        return code

    async def _execute_plot_code(
        self,
        code: str,
        output_path: Path,
    ) -> bool:
        """Execute Python code to generate figure using sandboxed execution.

        Args:
            code: Python code string
            output_path: Path to save output image

        Returns:
            True if successful, False otherwise
        """
        from autoforge.engine.sandbox import SubprocessSandbox

        # Replace placeholder with actual output path
        code = code.replace("{output_path}", str(output_path))

        # Write code into a temp working directory and execute in sandbox
        work_dir = output_path.parent / "_sandbox_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
        code_file = work_dir / "plot_code.py"
        code_file.write_text(code)

        sandbox = SubprocessSandbox(work_dir)
        try:
            await sandbox.start()
            result = await sandbox.exec(
                f"python plot_code.py",
                timeout=30,
            )
            await sandbox.stop()

            success = result.exit_code == 0 and output_path.exists()

            if not success and result.stderr:
                self.logger.warning(f"Execution error: {result.stderr}")

            return success
        except Exception as e:
            self.logger.error(f"Error executing code: {e}")
            try:
                await sandbox.stop()
            except Exception:
                pass
            return False
        finally:
            code_file.unlink(missing_ok=True)
            try:
                work_dir.rmdir()
            except OSError:
                pass

    def _fix_code_errors(self, code: str, error: str) -> str:
        """Attempt to fix code based on error message."""
        # Placeholder for iterative error fixing
        # In practice, this would call the LLM with the error
        return code


# ══════════════════════════════════════════════════════════════
# Figure Verification
# ══════════════════════════════════════════════════════════════


class FigureVerifier:
    """Verify figure claims against article text and reported metrics."""

    def __init__(self, llm: Any):
        """Initialize verifier with LLM client."""
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.FigureVerifier")

    async def verify_figure_claims(
        self,
        analysis: FigureAnalysis,
        article_claims: list[str],
    ) -> dict[str, Any]:
        """Verify if figure findings support article claims.

        Args:
            analysis: FigureAnalysis object
            article_claims: List of claims made in the article

        Returns:
            Verification report with support/contradiction assessment
        """
        results = []

        for claim in article_claims:
            prompt = f"""Does this figure support or contradict the following claim?

Claim: {claim}

Figure findings:
Description: {analysis.description}
Key findings: {analysis.key_findings}
Data points: {analysis.data_points}

Assess:
1. Does the figure support this claim? (yes/no/partial)
2. Confidence level (0-1)
3. Evidence from the figure
4. Any contradictions

Return as JSON with keys: supported, confidence, evidence, contradictions"""

            try:
                response = await self._call_llm_text(prompt)
                result = json.loads(response)
                results.append(
                    VerificationResult(
                        claim=claim,
                        supported=result.get("supported", False),
                        confidence=result.get("confidence", 0.0),
                        evidence=result.get("evidence", ""),
                        contradictions=result.get("contradictions", []),
                    )
                )
            except Exception as e:
                self.logger.error(f"Error verifying claim '{claim}': {e}")
                results.append(
                    VerificationResult(
                        claim=claim,
                        supported=False,
                        confidence=0.0,
                        evidence="",
                    )
                )

        return {
            "results": [r.__dict__ for r in results],
            "overall_support": sum(
                1 for r in results if r.supported
            ) / len(results) if results else 0.0,
        }

    async def cross_reference_data(
        self,
        analysis: FigureAnalysis,
        reported_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Cross-reference extracted data against reported metrics.

        Args:
            analysis: FigureAnalysis with extracted data
            reported_metrics: Metrics reported in article text

        Returns:
            Cross-reference report with consistency assessment
        """
        if not analysis.data_points or not reported_metrics:
            return {"matches": [], "discrepancies": []}

        prompt = f"""Compare these data extractions:

From figure:
{analysis.data_points}

Reported in article:
{reported_metrics}

Assess:
1. Which values match?
2. Which values differ and by how much?
3. Possible explanations for discrepancies
4. Data quality assessment

Return as JSON with keys: matches, discrepancies, explanations, quality"""

        try:
            response = await self._call_llm_text(prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error cross-referencing data: {e}")
            return {"matches": [], "discrepancies": []}

    async def _call_llm_text(self, prompt: str) -> str:
        """Call LLM with a text-only prompt via the LLM router."""
        if self.llm is None:
            raise RuntimeError("FigureVerifier requires an LLM router instance.")

        from autoforge.engine.llm_router import TaskComplexity

        messages = [{"role": "user", "content": prompt}]

        resp = await self.llm.call(
            complexity=TaskComplexity.STANDARD,
            system="You are an expert at verifying academic figure claims. Respond in valid JSON.",
            messages=messages,
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# Main Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════


class VLMFigurePipeline:
    """End-to-end orchestration of figure extraction, analysis, and verification."""

    def __init__(self, llm: Any):
        """Initialize pipeline with LLM client."""
        self.llm = llm
        self.extractor = FigureExtractor()
        self.analyzer = VLMAnalyzer(llm)
        self.reproducer = FigureReproducer(llm)
        self.verifier = FigureVerifier(llm)
        self.logger = logging.getLogger(f"{__name__}.VLMFigurePipeline")

    async def analyze_paper_figures(
        self,
        pdf_path: Path | None,
        figure_paths: list[Path],
        captions: list[str],
        output_dir: Path,
    ) -> list[FigureAnalysis]:
        """Analyze figures from a paper.

        Args:
            pdf_path: Path to PDF (optional, for extraction)
            figure_paths: List of extracted figure paths
            captions: Corresponding captions
            output_dir: Directory for output/reproductions

        Returns:
            List of FigureAnalysis objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract figures from PDF if provided
        if pdf_path and pdf_path.exists():
            metadata_list = await self.extractor.extract_figures(pdf_path)
            self.logger.info(f"Extracted {len(metadata_list)} figures from PDF")

        # Analyze provided figures
        if not figure_paths or not captions:
            return []

        # Ensure equal lengths
        min_len = min(len(figure_paths), len(captions))
        figure_paths = figure_paths[:min_len]
        captions = captions[:min_len]

        analyses = await self.analyzer.analyze_batch(
            list(zip(figure_paths, captions))
        )

        # Optionally reproduce figures
        repro_dir = output_dir / "reproductions"
        for analysis in analyses:
            repro_path = await self.reproducer.reproduce_figure(
                analysis, repro_dir
            )
            if repro_path:
                self.logger.info(
                    f"Reproduced {analysis.metadata.figure_id} at {repro_path}"
                )

        return analyses

    async def full_analysis(
        self,
        pdf_path: Path,
        article_text: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Perform complete analysis pipeline on a paper PDF.

        Args:
            pdf_path: Path to paper PDF
            article_text: Full text of article for context/verification
            output_dir: Directory for all outputs

        Returns:
            Comprehensive analysis report
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract figures from PDF
        metadata_list = await self.extractor.extract_figures(pdf_path)
        self.logger.info(f"Extracted {len(metadata_list)} figures")

        if not metadata_list:
            return {
                "pdf_path": str(pdf_path),
                "figures_found": 0,
                "analyses": [],
            }

        # Analyze each figure with article context
        analyses = []
        for metadata in metadata_list:
            if not metadata.source_path:
                continue

            # Extract figure image from PDF
            fig_image_path = await self._extract_figure_image(
                pdf_path, metadata, output_dir
            )
            if not fig_image_path:
                continue

            analysis = await self.analyzer.analyze_figure(
                fig_image_path, metadata.caption, context=article_text[:500]
            )
            analyses.append(analysis)

        # Extract claims from article for verification
        article_claims = self._extract_article_claims(article_text)

        # Verify figures against claims
        verification_results = []
        for analysis in analyses:
            result = await self.verifier.verify_figure_claims(
                analysis, article_claims
            )
            verification_results.append(result)

        # Generate report
        return {
            "pdf_path": str(pdf_path),
            "figures_found": len(metadata_list),
            "analyses": [a.to_dict() for a in analyses],
            "verifications": verification_results,
            "summary": {
                "total_figures": len(analyses),
                "average_confidence": (
                    sum(a.confidence for a in analyses) / len(analyses)
                    if analyses
                    else 0.0
                ),
            },
        }

    async def _extract_figure_image(
        self,
        pdf_path: Path,
        metadata: FigureMetadata,
        output_dir: Path,
    ) -> Path | None:
        """Extract a figure image from PDF."""
        if not metadata.bounding_box or not fitz:
            return None

        try:
            doc = fitz.open(pdf_path)
            try:
                page = doc[metadata.page_number - 1]

                # Render page region containing figure
                bbox = fitz.Rect(metadata.bounding_box)
                pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))

                output_path = output_dir / f"{metadata.figure_id}.png"
                pix.save(output_path)

                return output_path
            finally:
                doc.close()
        except Exception as e:
            self.logger.error(
                f"Error extracting figure image {metadata.figure_id}: {e}"
            )
            return None

    def _extract_article_claims(self, article_text: str) -> list[str]:
        """Extract key claims from article text."""
        # Simple heuristic: extract sentences containing metrics/numbers
        sentences = article_text.split(".")
        claims = [
            s.strip()
            for s in sentences
            if any(char.isdigit() for char in s)
        ]
        return claims[:10]  # Limit to first 10 for efficiency
