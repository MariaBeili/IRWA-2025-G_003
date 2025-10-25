import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re

from myapp.search.objects import Document


class ProcessedDocument(BaseModel):

    # Original
    _id: str
    pid: str
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

    # Processed
    title_processed: Optional[List[str]] = None
    description_processed: Optional[List[str]] = None
    brand_processed: Optional[List[str]] = None
    category_processed: Optional[List[str]] = None
    sub_category_processed: Optional[List[str]] = None
    seller_processed: Optional[List[str]] = None

    product_details_processed: Optional[Dict[str, Any]] = None

    search_text: Optional[List[str]] = None

    # --- CLASS METHODS ---

    @classmethod
    def from_document(cls, doc: Document) -> "ProcessedDocument":
        """
        Create a ProcessedDocument from a raw Document.
        It copies fields first, and you can later call process_fields().
        """
        return cls(
            _id=getattr(doc, "_id", None), # For some reason we don't get _id in Document correctly when loading the data
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
        Process all relevant fields for indexing.
        This calls smaller helper methods for modularity.
        """
        self.title_processed = self._process_text(self.title)
        self.description_processed = self._process_text(self.description)
        self.brand_processed = self._process_text(self.brand)
        self.category_processed = self._process_text(self.category)
        self.sub_category_processed = self._process_text(self.sub_category)
        self.seller_processed = self._process_text(self.seller)

        self.product_details_processed = self._process_product_details()

        self.search_text = self._combine_search_text()

    # --- HELPER METHODS ---

    def _process_text(self, text: Optional[str]) -> List[str]:
        """
        Tokenize, lowercase, remove stopwords/punctuation, and stem text.
        """
        if not text:
            return []

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

    def _process_product_details(self) -> Dict[str, Any]:
        """
        Process product_details into a dictionary of processed key values.
        Keys are not processed.
        """
        if not self.product_details:
            return {}

        processed_dict = {}

        for k, v in self.product_details.items():
            processed_value = self._process_text(str(v)) if v else None
            if k:
                processed_dict[k] = processed_value

        return processed_dict

    def _combine_search_text(self) -> List[str]:
        """Combine all relevant text fields into a single search text string."""
        parts = []
        if self.title_processed:
            parts += self.title_processed
        if self.description_processed:
            parts += self.description_processed
        if self.brand_processed:
            parts += self.brand_processed
        if self.category_processed:
            parts += self.category_processed
        if self.sub_category_processed:
            parts += self.sub_category_processed
        if self.seller_processed:
            parts += self.seller_processed
        if self.product_details_processed:
            for v in self.product_details_processed.values():
                parts += v

        return parts

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)