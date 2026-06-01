"""Converter registry: one converter per internal file type.

``get_converter(file_type)`` returns a shared converter instance, or ``None`` for an
unsupported type (the pipeline then marks the document ``skipped_unsupported``).
"""
from __future__ import annotations

from .base import BaseConverter, ConversionResult
from .csv_converter import CsvConverter
from .docx_converter import DocxConverter
from .excel_converter import ExcelConverter
from .html_converter import HtmlConverter
from .image_converter import ImageConverter
from .json_converter import JsonConverter
from .pdf_converter import PdfConverter
from .text_converter import TextConverter
from .xml_converter import XmlConverter

# Singleton instances (converters are stateless).
_REGISTRY: dict[str, BaseConverter] = {
    "pdf": PdfConverter(),
    "html": HtmlConverter(),
    "docx": DocxConverter(),
    "excel": ExcelConverter(),
    "csv": CsvConverter(),
    "json": JsonConverter(),
    "xml": XmlConverter(),
    "text": TextConverter(),
    "image": ImageConverter(),
}

SUPPORTED_TYPES = tuple(_REGISTRY.keys())


def get_converter(file_type: str | None) -> BaseConverter | None:
    """Return the converter for ``file_type`` or None if unsupported."""
    if not file_type:
        return None
    return _REGISTRY.get(file_type)


__all__ = [
    "BaseConverter",
    "ConversionResult",
    "get_converter",
    "SUPPORTED_TYPES",
]
