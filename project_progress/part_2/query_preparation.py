import unicodedata
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def process_query(line: str) -> list[str]:
    """
    Preprocess the article line (title + body) by:
    - Removing accents (é, ö, etc.)
    - Removing stop words
    - Stemming
    - Lowercasing
    - Tokenizing
    
    Argument:
    line -- string to be preprocessed

    Returns:
    tokens -- a list of tokens corresponding to the input line after preprocessing
    """

    if not line:
        return []
    
    line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("utf-8")

    # Lowercase
    line = line.lower()

    # Remove punctuation
    line = re.sub(r"[^\w\s]", " ", line)

    # Tokenize
    try:
        tokens = word_tokenize(line)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(line)

    # Loading stopwords
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens