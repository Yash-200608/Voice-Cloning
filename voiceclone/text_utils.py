import re

ABBREVIATIONS = {
    r"\bDr\.": "Doctor",
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Misses",
    r"\bMs\.": "Miss",
    r"\bSt\.": "Saint",
    r"\bvs\.": "versus",
    r"\betc\.": "etcetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
}

NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _expand_small_numbers(text):
    def repl(match):
        n = match.group(0)
        if len(n) == 1:
            return NUMBER_WORDS[n]
        return n
    return re.sub(r"\b\d\b", repl, text)


def normalize_text(text):
    text = text.strip()
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = _expand_small_numbers(text)
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text