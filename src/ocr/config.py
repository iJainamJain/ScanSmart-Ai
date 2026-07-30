"""Locate the Tesseract binary.

Tesseract is a separate executable rather than a Python package, so pytesseract
has to be told where it lives when it isn't on PATH. This lived inside the PDF
module, which meant anything else needing OCR silently found no Tesseract at
all; it belongs somewhere both callers can share.
"""

import os
import shutil
import sys

import pytesseract

# Checked in order; the first that exists wins.
WINDOWS_CANDIDATES = (
    r"D:\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def configure_tesseract() -> str | None:
    """Point pytesseract at a Tesseract binary; returns the path, or None."""
    on_path = shutil.which("tesseract")
    if on_path:
        pytesseract.pytesseract.tesseract_cmd = on_path
        return on_path

    if sys.platform == "win32":
        for candidate in WINDOWS_CANDIDATES:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return candidate
    return None


def tesseract_available() -> bool:
    """True if a usable Tesseract can actually be invoked."""
    configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


configure_tesseract()
