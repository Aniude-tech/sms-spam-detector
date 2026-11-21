import re
import html

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Must stay identical to the cleaning used in Notebook 3."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(http\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"\b\d{9,}\b", " phonetoken ", text)
    text = re.sub(r"[£$€]\s?\d[\d,\.]*", " currencytoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)