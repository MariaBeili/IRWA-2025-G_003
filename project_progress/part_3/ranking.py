import os
import sys
import math
import pickle
from json import JSONEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from myapp.search.load_corpus import load_corpus
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

from gensim.models import KeyedVectors
import numpy as np

# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/../../" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)

# We make sure to import from Part 2

from project_progress.part_2.query_preparation import process_query


# We load the index created in Part 2
INDEX_PATH = "project_progress/part_2/irwa_index.pkl"

def load_index():
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["tf"], data["idf"], data["title_index"]


# TF-IDF + COSINE SIMILARITY
def cosine_similarity(q_vec, d_vec):
    dot = 0.0
    for t in q_vec:
        dot += q_vec[t] * d_vec.get(t, 0.0)

    norm_q = math.sqrt(sum(v * v for v in q_vec.values()))
    norm_d = math.sqrt(sum(v * v for v in d_vec.values()))

    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


def rank_tfidf_cosine(query_terms, index, tf, idf):
    # AND semantics
    candidate_docs = None
    # We find the intersection of documents containing all query terms
    for t in query_terms:
        if t not in index:
            return []
        postings = set(pid for pid, _ in index[t])
        candidate_docs = postings if candidate_docs is None else candidate_docs.intersection(postings)
    # If no documents contain all terms, return empty list
    if not candidate_docs:
        return []

    # Query vector: TF = 1 for each term
    q_vec = {t: idf.get(t, 0.0) for t in query_terms}

    results = []

    # For each candidate document, calculate the tf-idf vector as in Part 2
    for pid in candidate_docs:
        d_vec = {}

        for term in query_terms:
            if term not in index:
                continue

            # Find the POSITION of the document within index[term]
            doc_index = None
            for i, (doc_id, _) in enumerate(index[term]):
                if doc_id == pid:
                    doc_index = i
                    break

            if doc_index is None:
                continue

            term_tf = tf[term][doc_index]  # same format as used in Part 2
            term_idf = idf.get(term, 0.0)

            d_vec[term] = float(term_tf) * float(term_idf)

        score = cosine_similarity(q_vec, d_vec)
        results.append((pid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# BM25 RANKING
# BM25 parameters
k1 = 1.5
b = 0.75

def compute_doc_len(pid, index):
    """Document length = total number of occurrences."""
    total = 0
    for term in index:
        for doc_id, positions in index[term]:
            if doc_id == pid:
                total += len(positions)
    return total


def compute_avg_doc_len(index):
    """Global average document length."""
    lengths = []
    seen = set()

    for term in index:
        for pid, positions in index[term]:
            if pid not in seen:
                seen.add(pid)
                lengths.append(len(positions))

    if not lengths:
        return 1.0
    return sum(lengths) / len(lengths)


def bm25_score(pid, query_terms, index, idf, avg_len):
    dl = compute_doc_len(pid, index)
    score = 0.0
    # For each term in the query, compute its BM25 contribution to the document score
    for term in query_terms:
        if term not in index:
            continue

        freq = 0
        for doc_id, positions in index[term]:
            if doc_id == pid:
                freq = len(positions)
                break
        # Skip if term does not appear in document
        if freq == 0:
            continue

        idf_term = idf.get(term, 0.0)

        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * (dl / avg_len))
        # BM25 formula
        score += idf_term * (numerator / denominator)

    return score

# Rank documents using BM25
def rank_bm25(query_terms, index, idf):
    candidate_docs = None
    # We find the intersection of documents containing all query terms
    for t in query_terms:
        if t not in index:
            return []
        postings = set(pid for pid, _ in index[t])
        candidate_docs = postings if candidate_docs is None else candidate_docs.intersection(postings)
    # If no documents contain all terms, return empty list
    if not candidate_docs:
        return []
    # Compute average document length
    avg_len = compute_avg_doc_len(index)
    results = []
    # For each candidate document, compute BM25 score
    for pid in candidate_docs:
        score = bm25_score(pid, query_terms, index, idf, avg_len)
        results.append((pid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results



# El de casa
def cosine_numeric(query_score, query_price, doc_score, doc_price):
    dot = query_score * doc_score + query_price * doc_price

    norm_q = math.sqrt(query_score ** 2 + query_price ** 2)
    norm_d = math.sqrt(doc_score ** 2 + doc_price ** 2)

    if norm_q == 0 or norm_d == 0:
        return 0.0

    return dot / (norm_q * norm_d)


def rank_custom_cosine(query_terms, index, tf, idf, corpus, query_score, query_price):
    """
    corpus = dict[pid] -> Document
    query_score (float)
    query_price (float)
    """

    # AND semantics same as your other functions
    candidate_docs = None
    for t in query_terms:
        if t not in index:
            return []
        postings = set(pid for pid, _ in index[t])
        candidate_docs = postings if candidate_docs is None else candidate_docs.intersection(postings)
    if not candidate_docs:
        return []

    # Build query TF-IDF vector
    q_vec = {t: idf.get(t, 0.0) for t in query_terms}

    results = []

    for pid in candidate_docs:

        # Build document TF-IDF vector (same method as in TF-IDF ranker)
        d_vec = {}
        for term in query_terms:
            if term not in index:
                continue

            # find doc in postings
            doc_index = None
            for i, (doc_id, _) in enumerate(index[term]):
                if doc_id == pid:
                    doc_index = i
                    break

            if doc_index is None:
                continue

            term_tf = tf[term][doc_index]
            term_idf = idf.get(term, 0.0)
            d_vec[term] = float(term_tf) * float(term_idf)

        # TEXT COSINE
        text_cos = cosine_similarity(q_vec, d_vec)

        # NUMERIC COSINE based on Document metadata
        doc = corpus[pid]

        numeric_cos = cosine_numeric(
            query_score,
            query_price,
            doc.average_rating or 0.0,
            doc.selling_price or 0.0
        )

        # FINAL SCORE = 50/50
        final_score = 0.5 * text_cos + 0.5 * numeric_cos

        results.append((pid, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def cosine_similarity_vec(q_vec, d_vec):
    if q_vec is None or d_vec is None:
        return 0.0
    dot = np.dot(q_vec, d_vec)
    norm_q = np.linalg.norm(q_vec)
    norm_d = np.linalg.norm(d_vec)
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)

def text_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

def rank_word2vec_cos(query_terms, doc_vectors, model):
    q_vec = text_vector(query_terms, model)
    scores = [(pid, cosine_similarity_vec(q_vec, d_vec)) for pid, d_vec in doc_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:20]

# Main 
QUERIES = [
    "ARBO cotton track pants for men",
    "Multicolor track pants combo ECKO",
    "Black solid women track pants",
    "Elastic waist cotton blend track pants",
    "Self design multicolor track pants"
]

PRICES = [
    50.0,
    30.0,
    20.0,
    15.0,
    30.0
]

RATINGS = [
    4.0,
    3.0,
    2.0,
    4.0,
    1.0
]

OUTPUT_FILE = "project_progress/part_3/ranking_results.txt"


def main():

    model = KeyedVectors.load_word2vec_format("resources/GoogleNews-vectors-negative300.bin", binary=True)

    index, tf, idf, title_index = load_index()

    doc_vectors = {}
    for pid, doc in corpus.items():
        text_parts = [getattr(doc,'title',''), getattr(doc,'description','')]
        text = ' '.join(filter(None,text_parts))
        doc_vectors[pid] = text_vector(text, model)

    os.makedirs("project_progress/part_3", exist_ok=True)
    f = open(OUTPUT_FILE, "w", encoding="utf-8")

    print("\nRANKING RESULTS (TF-IDF + COSINE & BM25)\n")
    f.write("RANKING RESULTS (TF-IDF + COSINE & BM25)\n\n")

    for query, price, rating in zip(QUERIES, PRICES, RATINGS):
        processed = process_query(query)

        print(f"\nQuery: {query}")
        print(f"Processed: {processed}")
        f.write(f"\nQuery: {query}\n")
        f.write(f"Processed: {processed}\n")

        # TF-IDF + Cosine
        tfidf_results = rank_tfidf_cosine(processed, index, tf, idf)

        print("\nTop 5 (TF-IDF + Cosine):")
        f.write("\nTop 5 (TF-IDF + Cosine):\n")

        for pid, score in tfidf_results[:5]:
            title = title_index.get(pid, "Unknown Title")
            print(f"  {pid} | {score:.4f} | {title}")
            f.write(f"  {pid} | {score:.4f} | {title}\n")

        # BM25
        bm25_results = rank_bm25(processed, index, idf)

        print("\nTop 5 (BM25):")
        f.write("\nTop 5 (BM25):\n")

        for pid, score in bm25_results[:5]:
            title = title_index.get(pid, "Unknown Title")
            print(f"  {pid} | {score:.4f} | {title}")
            f.write(f"  {pid} | {score:.4f} | {title}\n")

        # El de casa
        custom_results = rank_custom_cosine(processed, index, tf, idf, corpus, rating, price)

        print("\nTop 5 (custom):")
        f.write("\nTop 5 (custom):\n")

        for pid, score in custom_results[:5]:
            title = title_index.get(pid, "Unknown Title")
            print(f"  {pid} | {score:.4f} | {title}")
            f.write(f"  {pid} | {score:.4f} | {title}\n")

        # word2vec + cosine

        word2vec_cosine_results = rank_word2vec_cos(query, doc_vectors, model)

        print("\nTop 20 word2vec + cosine:")
        f.write("\nTop 20 word2vec + cosine:\n")
        for pid,score in word2vec_cosine_results:
            title = title_index.get(pid,'Unknown Title')
            print(f'{pid} | {score:.4f} | {title}')
            f.write(f'{pid} | {score:.4f} | {title}\n')

        print("-" * 60)
        f.write("-" * 60 + "\n")

    f.close()
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()