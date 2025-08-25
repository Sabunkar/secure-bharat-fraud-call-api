import re
import math
import logging

logger = logging.getLogger(__name__)

def normalize_number(number) -> str:
    """Return E.164-ish string: keep only leading + and digits."""
    try:
        number = str(number).strip()
        cleaned = re.sub(r"[^\d+]", "", number)
        if cleaned.startswith("+"):
            return "+" + re.sub(r"\D", "", cleaned[1:])
        return "+" + re.sub(r"\D", "", cleaned)
    except Exception as e:
        logger.error(f"Error normalizing number {number}: {e}")
        return "+0"  # Fallback

def shannon_entropy(d: str) -> float:
    """Calculate Shannon entropy of a string."""
    try:
        if not d:
            return 0.0
        totals = len(d)
        if totals == 0:
            return 0.0
        
        # Count frequency of each character
        char_counts = {}
        for ch in d:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        
        # Calculate probabilities and entropy
        entropy = 0.0
        for count in char_counts.values():
            prob = count / totals
            if prob > 0:
                entropy -= prob * math.log(prob, 2)
        
        return entropy
    except Exception as e:
        logger.error(f"Error calculating entropy for {d}: {e}")
        return 0.0

def extract_features(number: str) -> dict:
    """Extract features from a phone number."""
    try:
        num = normalize_number(number)
        digits = re.sub(r"\D", "", num)
        
        # Basic features
        length = len(digits)
        
        # Country code extraction
        cc = 0
        if length >= 2:
            try:
                cc = int(digits[:2])
            except ValueError:
                cc = 0
        
        # Special case for US/Canada numbers
        if digits.startswith("1") and length in (10, 11):
            cc = 1
        
        # Starting digits
        starts_with = 0
        if length >= 3:
            try:
                starts_with = int(digits[:3])
            except ValueError:
                starts_with = 0
        
        # Unique digit ratio
        ratio_unique = 0.0
        if length > 0:
            unique_digits = len(set(digits))
            ratio_unique = unique_digits / length
        
        # Detect runs of 3 or more same digits
        runs_ge3 = 0
        if re.search(r"(\d)\1{2,}", digits):
            runs_ge3 = 1
        
        # Calculate entropy
        entropy_proxy = shannon_entropy(digits)
        
        # Pattern detection
        patterns = {
            "has_000": 1 if "000" in digits else 0,
            "has_111": 1 if "111" in digits else 0,
            "has_123": 1 if "123" in digits else 0,
            "has_987": 1 if "987" in digits else 0,
            "has_555": 1 if "555" in digits else 0,
        }
        
        # Combine all features
        feats = {
            "length": length,
            "country_code": cc,
            "starts_with": starts_with,
            "ratio_unique": ratio_unique,
            "runs_ge3": runs_ge3,
            "entropy_proxy": entropy_proxy,
            **patterns
        }
        
        # Validate all features are numeric
        for key, value in feats.items():
            if not isinstance(value, (int, float)):
                logger.warning(f"Non-numeric feature {key}: {value}, converting to 0")
                feats[key] = 0
            elif math.isnan(value) or math.isinf(value):
                logger.warning(f"Invalid numeric value for {key}: {value}, converting to 0")
                feats[key] = 0
        
        return feats
        
    except Exception as e:
        logger.error(f"Error extracting features from {number}: {e}")
        # Return default safe features
        return {
            "length": 0,
            "country_code": 0,
            "starts_with": 0,
            "ratio_unique": 0.0,
            "runs_ge3": 0,
            "entropy_proxy": 0.0,
            "has_000": 0,
            "has_111": 0,
            "has_123": 0,
            "has_987": 0,
            "has_555": 0,
        }

# Test function
def test_extract_features():
    """Test feature extraction with sample numbers."""
    test_numbers = [
        "+911234567890",
        "+919876543210", 
        "+14155552671",
        "911100001111",
        "+1-800-555-1234"
    ]
    
    for number in test_numbers:
        feats = extract_features(number)
        print(f"Number: {number}")
        print(f"Features: {feats}")
        print("-" * 50)

if __name__ == "__main__":
    test_extract_features()