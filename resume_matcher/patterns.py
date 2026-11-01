"""Regex helpers for years of experience, dates, degrees and contact details."""

from __future__ import annotations

import re
from datetime import date

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
}
_NUM = r"(\d{1,2}(?:\.\d)?|" + "|".join(_WORD_NUMBERS) + r")"

# "3+ years", "2-4 years", "5 or more years", "three years", "at least 2 yrs"
YEARS_PATTERN = re.compile(
    rf"\b{_NUM}\s*(?:\+|plus)?\s*(?:(?:-|to|or)\s*(?:{_NUM}|more)\s*\+?\s*)?(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_EXPERIENCE_WORD = re.compile(r"\b(?:experience|exp|industry|professional|working|work)\b", re.IGNORECASE)


def _to_number(token: str) -> float:
    token = token.lower()
    return float(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else float(token)


def find_years_requirements(text: str) -> list[float]:
    """Minimum years from statements like '3+ years of experience' (lower bound of ranges)."""
    values: list[float] = []
    for match in YEARS_PATTERN.finditer(text):
        tail = text[match.end() : match.end() + 60]
        head = text[max(0, match.start() - 40) : match.start()]
        if _EXPERIENCE_WORD.search(tail) or _EXPERIENCE_WORD.search(head):
            values.append(_to_number(match.group(1)))
    return values


# ------------------------------------------------------------------ date ranges

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)


def _date(suffix: str) -> str:
    return (
        rf"(?:(?P<m{suffix}>{_MONTH})\s*,?\s*|(?P<n{suffix}>\d{{1,2}})\s*[/.-]\s*)?"
        rf"(?P<y{suffix}>(?:19|20)\d{{2}})"
    )


DATE_RANGE_PATTERN = re.compile(
    rf"{_date('1')}\s*(?:-|to|until|till)\s*(?:{_date('2')}|(?P<current>present|current|now|today|ongoing|date))",
    re.IGNORECASE,
)


def _month_index(month_name: str | None, month_num: str | None) -> int | None:
    if month_name:
        return _MONTHS[month_name.lower()[:3]]
    if month_num and 1 <= int(month_num) <= 12:
        return int(month_num)
    return None


def find_date_ranges(text: str, today: date) -> list[tuple[int, int]]:
    """Employment intervals as (start_month_index, end_month_index), month index = year*12+month."""
    intervals: list[tuple[int, int]] = []
    now_index = today.year * 12 + today.month
    for m in DATE_RANGE_PATTERN.finditer(text):
        start_month = _month_index(m.group("m1"), m.group("n1"))
        start = int(m.group("y1")) * 12 + (start_month or 1)
        if m.group("current"):
            end = now_index
            months_given = start_month is not None
        else:
            end_month = _month_index(m.group("m2"), m.group("n2"))
            end = int(m.group("y2")) * 12 + (end_month or 1)
            months_given = start_month is not None and end_month is not None
        if months_given:
            end += 1  # "Jan 2020 - Jan 2020" is one month
        if end <= start:
            end = start + 6 if not months_given and end == start else end
        if start > now_index or end <= start:
            continue
        intervals.append((start, min(end, now_index + 1)))
    return intervals


def total_months(intervals: list[tuple[int, int]]) -> int:
    """Total months covered, merging overlapping intervals."""
    total = 0
    current_start: int | None = None
    current_end = 0
    for start, end in sorted(intervals):
        if current_start is None or start > current_end:
            if current_start is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


# ------------------------------------------------------------------ education

EDUCATION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "phd": [
        re.compile(r"\bph\.?\s?d\b\.?", re.IGNORECASE),
        re.compile(r"\bdoctor(?:ate|al)\b", re.IGNORECASE),
    ],
    "master": [
        re.compile(r"\bmaster'?s?\b(?=\s+(?:degree|of|in)\b)", re.IGNORECASE),
        re.compile(r"\bmaster'?s\b(?!\s+(?:class|card))", re.IGNORECASE),
        re.compile(r"(?<![A-Za-z])M\.\s?(?:S|Sc|Tech|E|Eng|A)\.?(?![A-Za-z])"),
        re.compile(r"\b(?:MSc|MTech|MEng|MBA|MCA)\b"),
        re.compile(r"\bMS\b(?=\s+(?:in|degree)\b)"),
        re.compile(r"\bpostgraduate\b|\bpost-graduate\b", re.IGNORECASE),
    ],
    "bachelor": [
        re.compile(r"\bbachelor'?s?\b", re.IGNORECASE),
        re.compile(r"(?<![A-Za-z])B\.\s?(?:S|Sc|Tech|E|Eng|A|Com)\.?(?![A-Za-z])"),
        re.compile(r"\b(?:BSc|BTech|BEng|BCA|BBA|BCom)\b"),
        re.compile(r"\bBS\b(?=\s+(?:in|degree)\b)"),
        re.compile(r"\bundergraduate degree\b", re.IGNORECASE),
    ],
    "diploma": [re.compile(r"\bdiploma\b|\bassociate'?s? degree\b", re.IGNORECASE)],
    "high_school": [re.compile(r"\bhigh school\b|\bsecondary school\b|\bhsc\b|\bssc\b", re.IGNORECASE)],
}


def find_education_levels(text: str) -> list[str]:
    return [level for level, patterns in EDUCATION_PATTERNS.items() if any(p.search(text) for p in patterns)]


# ------------------------------------------------------------------ contact / PII

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"\b(?:19|20)\d{2}\b")
QUANTIFIER_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|x\b|k\b|m\b|ms\b|users|customers|hours|days|weeks|people|members|"
    r"requests|records|rows|models|pipelines|projects|students)|[$€£₹]\s?\d",
    re.IGNORECASE,
)


def looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15:
        return False
    # Avoid treating date ranges such as "2019 - 2021" as phone numbers.
    return not re.fullmatch(r"\s*(?:19|20)\d{2}\s*-\s*(?:19|20)\d{2}\s*", candidate)


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[email]", text)
    text = URL_PATTERN.sub("[link]", text)
    return PHONE_PATTERN.sub(lambda m: "[phone]" if looks_like_phone(m.group(0)) else m.group(0), text)


def is_quantified(line: str) -> bool:
    """True if the line contains a measurable result (numbers that are not just years)."""
    if QUANTIFIER_PATTERN.search(line):
        return True
    without_years = _YEAR_ONLY.sub("", line)
    return bool(re.search(r"\b\d{2,}\b", without_years))
