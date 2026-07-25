import re

def clean_arabic_text(text: str) -> str:
    if not text:
        return ""
    diacritic_pattern = re.compile(r'[\u064B-\u0652]')
    text = diacritic_pattern.sub('', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    sample = "السَّلامُ عَلَيكُم وَرَحمَةُ اللهِ وَبَرَكَاتُه"
    print("Original:", sample)
    print("Cleaned:", clean_arabic_text(sample))
