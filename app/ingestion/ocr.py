"""Optional OCR for scanned PDF pages (disabled by default).

Requires `pip install -r requirements-ocr.txt` and the Tesseract binary.
"""

from __future__ import annotations

import io


class OCRUnavailableError(RuntimeError):
    pass


def ocr_pdf_page(page, tesseract_cmd: str | None = None, lang: str = "eng", dpi: int = 300) -> str:
    """Render a PyMuPDF page to an image and run Tesseract on it."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise OCRUnavailableError(
            "OCR is enabled but pytesseract/Pillow are not installed. "
            "Run: pip install -r requirements-ocr.txt"
        ) from exc
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractNotFoundError as exc:  # pragma: no cover
        raise OCRUnavailableError(
            "Tesseract binary not found. Install it and set TESSERACT_CMD in .env."
        ) from exc
