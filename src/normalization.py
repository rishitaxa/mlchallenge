import re
import unicodedata
import pandas as pd
from typing import List, Set, Tuple
RE_UNICODE_COMBINING = re.compile(r'[\u0300-\u036f]')
RE_AMPERSAND = re.compile(r'\b&\b|&')
RE_NON_ALPHANUMERIC = re.compile(r'[^a-z0-9\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_NUMERIC_TOKENS = re.compile(r'\b\d+[a-z]?\b')
RE_POSTAL_CODE = re.compile(r'\b\d{4,8}\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b')
LEGAL_SUFFIXES = {
    "inc": "", "incorporated": "", "ltd": "", "limited": "",
    "llc": "", "lc": "", "corp": "", "corporation": "",
    "gmbh": "", "plc": "", "co": "company", "company": "company",
    "pvt": "private", "private": "private", "sa": "", "bv": "",
    "nv": "", "sarl": "", "srl": "", "sl": "", "ag": "", "spa": "",
    "oy": "","ab": "", "as": "", "pt": "", "tbk": "", "ltd.": "", "inc.": "",
    "co.": "company", "corp.": ""
}
NAME_ABBREVIATIONS = {
    "mgmt": "management",
    "mgt": "management",
    "intl": "international",
    "internatl": "international",
    "tech": "technology",
    "technologies": "technology",
    "technol": "technology",
    "svcs": "services",
    "svc": "services",
    "serv": "services",
    "solns": "solutions",
    "soln": "solutions",
    "solut": "solutions",
    "sys": "systems",
    "system": "systems",
    "ind": "industries",
    "inds": "industries",
    "mfg": "manufacturing",
    "manuf": "manufacturing",
    "dev": "development",
    "grp": "group",
    "eng": "engineering",
    "engr": "engineering",
    "assoc": "associates",
    "assn": "association",
    "dist": "distribution",
    "distrib": "distribution",
    "prod": "products",
    "prods": "products"
}
ADDRESS_ABBREVIATIONS = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue", "av": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "ct": "court", "ct.": "court",
    "pl": "place", "pl.": "place",
    "sq": "square", "sq.": "square",
    "hwy": "highway", "hwy.": "highway",
    "pkwy": "parkway", "pkwy.": "parkway",
    "str": "strasse", "str.": "strasse",
    "ste": "suite", "ste.": "suite",
    "apt": "apartment", "apt.": "apartment",
    "bldg": "building", "bldg.": "building",
    "fl": "floor", "fl.": "floor",
    "rm": "room", "rm.": "room",
    "po box": "pobox", "p.o. box": "pobox", "p.o.box": "pobox",
    "post office box": "pobox",
    "no": "number", "no.": "number", "#": "number"
}
def strip_unicode(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))
def normalize_country(country: str) -> str:
    if not country:
        return ""
    text = str(country).lower().strip()
    text = strip_unicode(text)
    text = RE_NON_ALPHANUMERIC.sub('', text)
    return text.strip()
def normalize_business_name(name: str) -> str:
    if not name:
        return ""
    text = str(name).lower()
    text = strip_unicode(text)
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUMERIC.sub(' ', text)
    tokens = text.split()
    norm_tokens = []
    for token in tokens:
        if token in LEGAL_SUFFIXES:
            sub = LEGAL_SUFFIXES[token]
            if sub:
                norm_tokens.append(sub)
        elif token in NAME_ABBREVIATIONS:
            norm_tokens.append(NAME_ABBREVIATIONS[token])
        else:
            norm_tokens.append(token)
    result = " ".join(norm_tokens)
    return RE_MULTI_SPACE.sub(' ', result).strip()
def normalize_business_address(address: str) -> str:
    if not address:
        return ""
    text = str(address).lower()
    text = strip_unicode(text)
    for raw_pobox in ["p.o. box", "po box", "post office box"]:
        text = text.replace(raw_pobox, "pobox")
    text = RE_NON_ALPHANUMERIC.sub(' ', text)
    tokens = text.split()
    norm_tokens = []
    for token in tokens:
        if token in ADDRESS_ABBREVIATIONS:
            sub = ADDRESS_ABBREVIATIONS[token]
            if sub:
                norm_tokens.append(sub)
        else:
            norm_tokens.append(token)
    result = " ".join(norm_tokens)
    return RE_MULTI_SPACE.sub(' ', result).strip()
def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    return RE_NUMERIC_TOKENS.findall(text)
def extract_postal_code(address: str) -> str:
    if not address:
        return ""
    matches = RE_POSTAL_CODE.findall(address)
    return matches[-1] if matches else ""
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["business_name"] = df.get("business_name", "").fillna("").astype(str)
    df["business_address"] = df.get("business_address", "").fillna("").astype(str)
    df["country"] = df.get("country", "").fillna("").astype(str)
    df["norm_country"] = df["country"].apply(normalize_country)
    df["norm_name"] = df["business_name"].apply(normalize_business_name)
    df["norm_address"] = df["business_address"].apply(normalize_business_address)
    df["name_tokens"] = df["norm_name"].apply(lambda x: [t for t in x.split() if t])
    df["address_tokens"] = df["norm_address"].apply(lambda x: [t for t in x.split() if t])
    df["numeric_tokens"] = (df["norm_name"] + " " + df["norm_address"]).apply(extract_numeric_tokens)
    df["postal_code"] = df["norm_address"].apply(extract_postal_code)
    return df
