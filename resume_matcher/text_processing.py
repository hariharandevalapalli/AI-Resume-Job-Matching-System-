"""Text normalisation, line handling and resume section detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

GENERAL_SECTION = "general"

_BULLETS = "•●▪◦■□‣⁃∙➢✓✔·"
_BULLET_PREFIX = re.compile(rf"^\s*(?:[{_BULLETS}]|[-*>]|\d{{1,2}}[.)])\s+")
_DASHES = re.compile("[‐‑‒–—―−]")
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}
_MULTI_SPACE = re.compile(r"[ \t ]+")
_SENTENCE_END = re.compile(r"[.!?:;]$")


def normalize_text(text: str) -> str:
    """Unicode-normalise, unify dashes/quotes, drop bullets and collapse spaces, keeping lines."""
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    for src, dst in _QUOTES.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        line = _BULLET_PREFIX.sub("", raw)
        line = line.strip(f" \t{_BULLETS}")
        line = _MULTI_SPACE.sub(" ", line)
        lines.append(line)
    collapsed = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def merge_wrapped_lines(lines: list[str]) -> list[str]:
    """Re-join lines that a PDF wrapped mid-sentence (next line starts lowercase)."""
    merged: list[str] = []
    for line in lines:
        if not line:
            continue
        if merged and not _SENTENCE_END.search(merged[-1]) and line[0].islower() and len(merged[-1].split()) >= 4:
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return merged


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _header_key(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z ]", " ", text)
    return " ".join(text.split())


@dataclass
class Sections:
    """Resume lines grouped by section, in document order."""

    lines: dict[str, list[str]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, section: str, line: str) -> None:
        if section not in self.lines:
            self.lines[section] = []
            self.order.append(section)
        self.lines[section].append(line)

    def text(self, section: str) -> str:
        return "\n".join(self.lines.get(section, []))

    def found(self) -> list[str]:
        """Recognised sections (excludes the implicit 'general' bucket)."""
        return [s for s in self.order if s != GENERAL_SECTION]

    def items(self) -> list[tuple[str, list[str]]]:
        return [(s, self.lines[s]) for s in self.order]


class SectionDetector:
    """Detects resume section headers such as 'Experience' or 'Technical Skills:'."""

    def __init__(self, section_aliases: dict[str, list[str]]) -> None:
        self._lookup: dict[str, str] = {}
        for section, aliases in section_aliases.items():
            for alias in [section, *aliases]:
                self._lookup[_header_key(alias)] = section

    def match_header(self, line: str) -> tuple[str, str] | None:
        """Return (section, remainder) if the line is a header, else None.

        Handles stand-alone headers ("EXPERIENCE") and inline headers
        ("Skills: Python, SQL") where the remainder is the content after the colon.
        """
        stripped = line.strip()
        if not stripped:
            return None
        if ":" in stripped:
            head, _, rest = stripped.partition(":")
            section = self._lookup.get(_header_key(head))
            if section and len(head.split()) <= 5:
                return section, rest.strip()
        if len(stripped.split()) <= 5 and not stripped.endswith("."):
            section = self._lookup.get(_header_key(stripped))
            if section:
                return section, ""
        return None

    def split(self, text: str) -> Sections:
        sections = Sections()
        current = GENERAL_SECTION
        raw_lines = [ln for ln in text.split("\n") if ln.strip()]
        for line in raw_lines:
            header = self.match_header(line)
            if header:
                current, remainder = header
                if remainder:
                    sections.add(current, remainder)
                elif current not in sections.lines:
                    sections.lines[current] = []
                    sections.order.append(current)
                continue
            sections.add(current, line)
        for name in sections.order:
            sections.lines[name] = merge_wrapped_lines(sections.lines[name])
        return sections


def snippet_around(line: str, start: int, end: int, width: int = 220) -> str:
    """Trim a long line to roughly `width` characters centred on [start, end)."""
    if len(line) <= width:
        return line
    pad = max(0, (width - (end - start)) // 2)
    lo = max(0, start - pad)
    hi = min(len(line), end + pad)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(line) else ""
    return f"{prefix}{line[lo:hi].strip()}{suffix}"
