import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re
import unicodedata

from myapp.search.objects import Document


class ProcessedDocument(BaseModel):
    _id: str
    pid: str

    # Original (for display/filter)
    title: str
    description: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    sub_category: Optional[str]
    product_details: Optional[Dict[str, Any]]
    seller: Optional[str]
    out_of_stock: bool
    selling_price: Optional[float]
    discount: Optional[float]
    actual_price: Optional[float]
    average_rating: Optional[float]
    url: Optional[str]

    # Preprocessed
    title_processed: Optional[List[str]] = None
    description_processed: Optional[List[str]] = None
    brand_normalized: Optional[str] = None
    category_normalized: Optional[str] = None
    sub_category_normalized: Optional[Dict[str, Any]] = None
    product_details_processed: Optional[List[str]] = None
    seller_normalized: Optional[str] = None
    search_text: Optional[List[str]] = None

    # --- CLASS METHODS ---

    @classmethod
    def from_document(cls, doc: Document) -> "ProcessedDocument":
        """
        Create a ProcessedDocument from a raw Document.
        It copies fields first, and you can later call process_fields().
        """
        return cls(
            _id=getattr(doc, "_id", None),
            pid=doc.pid,
            title=doc.title,
            description=doc.description,
            brand=doc.brand,
            category=doc.category,
            sub_category=doc.sub_category,
            product_details=doc.product_details,
            seller=doc.seller,
            out_of_stock=doc.out_of_stock,
            selling_price=doc.selling_price,
            discount=doc.discount,
            actual_price=doc.actual_price,
            average_rating=doc.average_rating,
            url=doc.url
        )

    def process_fields(self):
        """
        Preprocess all relevant fields for indexing.
        This calls smaller helper methods for modularity.
        """
        self.title_processed = self._preprocess_text(self.title)
        self.description_processed = self._preprocess_text(self.description)

        self.brand_normalized = self._normalize_category_field(self.brand)
        self.category_normalized = self._normalize_category_field(self.category)
        self.sub_category_normalized = self._normalize_category_field(self.sub_category)
        self.seller_normalized = self._normalize_category_field(self.seller)

        self.product_details_processed = self._process_product_details()

        self.search_text = self._combine_search_text()

    # --- HELPER METHODS ---

    def _preprocess_text(self, text: Optional[str]) -> List[str]:
        """Tokenize, lowercase, remove stopwords/punctuation, and stem text."""
        if not text:
            return []

        # Normalize accents
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            tokens = word_tokenize(text)

        # Remove stopwords
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
            
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

        return tokens

    def _normalize_category_field(self, value: Optional[str]) -> Optional[str]:
        """Simple normalization for categorical/keyword fields."""
        if not value:
            return None
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8", "ignore")
        return value.strip().lower()

    def _process_product_details(self) -> Dict[str, Optional[str]]:
        """
        Process product_details into a dictionary of normalized key-value pairs.
        Example:
        [{"Type": "Round Neck"}, {"Sleeve": "Half Sleeve"}]
        → {"type": "round neck", "sleeve": "half sleeve"}
        """
        if not self.product_details:
            return {}

        processed = {}

        for k, v in self.product_details.items():
            norm_key = self._normalize_category_field(str(k)) if k else None
            norm_value = self._normalize_category_field(str(v)) if v else None
            if norm_key:
                processed[norm_key] = norm_value

        return processed

    def _combine_search_text(self) -> List[str]:
        """Combine all relevant text fields into a single search text string."""
        parts = []
        if self.title_processed:
            parts += self.title_processed
        if self.description_processed:
            parts += self.description_processed
        if self.brand_normalized:
            parts.append(self.brand_normalized)
        if self.category_normalized:
            parts.append(self.category_normalized)
        if self.sub_category_normalized:
            parts.append(self.sub_category_normalized)
        if self.product_details_processed:
            parts += [v for v in self.product_details_processed.values() if v]
        if self.seller_normalized:
            parts.append(self.seller_normalized)

        return self._preprocess_text(" ".join(parts).strip())

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)