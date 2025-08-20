import re, math

def normalize_number(number) -> str:
    """Return E.164-ish string: keep only leading + and digits."""
    number = str(number).strip()
    cleaned = re.sub(r"[^\d+]", "", number)
    if cleaned.startswith("+"):
        return "+" + re.sub(r"\D", "", cleaned[1:])
    return "+" + re.sub(r"\D", "", cleaned)

def shannon_entropy(d: str) -> float:
    if not d:
        return 0.0
    totals = len(d)
    probs = [d.count(ch)/totals for ch in set(d)]
    return -sum(p * math.log(p, 2) for p in probs)

def extract_features(number: str) -> dict:
    num = normalize_number(number)
    digits = re.sub(r"\D", "", num)

    length = len(digits)
    cc = int(digits[:2]) if length >= 2 else 0
    if digits.startswith("1") and length in (10,11):
        cc = 1
    starts_with = int(digits[:3]) if length >= 3 else 0

    runs_ge3 = 1 if re.search(r"(\d)\1{2,}", digits) else 0

    feats = {
        "length": length,
        "country_code": cc,
        "starts_with": starts_with,
        "ratio_unique": (len(set(digits)) / length) if length else 0.0,
        "runs_ge3": runs_ge3,
        "entropy_proxy": shannon_entropy(digits),
        "has_000": 1 if "000" in digits else 0,
        "has_111": 1 if "111" in digits else 0,
        "has_123": 1 if "123" in digits else 0,
        "has_987": 1 if "987" in digits else 0,
        "has_555": 1 if "555" in digits else 0,
    }
    return feats
