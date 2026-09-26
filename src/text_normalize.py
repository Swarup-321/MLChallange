"""
Turns messy raw strings ("Sharma & Sons Pvt. Ltd.") into a clean, comparable
form ("sharma and sons"). Every other module imports from here so that
blocking and feature-building always see text normalized the exact same way.
"""
import re

# Legal-entity suffixes we strip out because they carry almost no
# identifying signal (every third business has "Ltd" in its name) and
# just add noise to token-overlap comparisons.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "pvt", "private", "plc", "gmbh", "llp", "lp",
    "pllc", "sa", "sarl", "srl", "bv", "nv", "ag", "kg", "oy", "as",
}

# Common abbreviation -> expansion pairs for both names and addresses.
# Expanding (rather than abbreviating) is the safer direction because it
# only ever adds information, never throws it away.
ABBREV_MAP = {
    "&": "and",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "mgmt": "management",
    "intl": "international",
    "svcs": "services",
    "svc": "service",
    "mfg": "manufacturing",
    "assoc": "associates",
    "dept": "department",
    "govt": "government",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def normalize_text(text) -> str:
    """Lowercase, drop punctuation, expand abbreviations. Returns '' for NaN."""
    if text is None or (isinstance(text, float) and text != text):  # NaN check
        return ""
    tokens = _tokenize(str(text))
    tokens = [ABBREV_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


def normalize_name(text) -> str:
    """Normalize a business name AND strip legal suffixes (for blocking/matching)."""
    norm = normalize_text(text)
    tokens = [t for t in norm.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens) if tokens else norm  # never return empty if we had *something*


def name_tokens(text) -> set:
    return set(normalize_name(text).split())


def address_tokens(text) -> set:
    return set(normalize_text(text).split())


def extract_postal_code(address) -> str:
    """Grab a bare 5-6 digit run (US ZIP / India PIN) if one exists, else ''."""
    if address is None:
        return ""
    m = re.search(r"\b(\d{5,6})\b", str(address))
    return m.group(1) if m else ""
