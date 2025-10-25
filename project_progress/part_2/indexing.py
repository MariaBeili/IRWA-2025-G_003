from collections import defaultdict
from array import array
import numpy as np
import math
import collections
from collections import defaultdict
import numpy.linalg as la

from myapp.search.objects import Document
from project_progress.part_1.data_preparation import ProcessedDocument
from project_progress.part_2.query_preparation import process_query


def create_index_tfidf(documents: list[ProcessedDocument]):
    """
    Implement the inverted index and compute tf, df, and idf for the corpus.

    Arguments:
    documents -- collection of raw Document objects

    Returns:
    index - the inverted index: term -> list of [doc_id, [positions]]
    tf - normalized term frequency for each term in each document
    df - number of documents each term appears in
    idf - inverse document frequency of each term
    title_index - mapping of pid -> title
    """
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)
    title_index = defaultdict(str)

    num_documents = len(documents)


    for doc in documents:
        # Store title for lookup
        title_index[doc.pid] = doc.title

        # Build term positions map for this document
        current_doc_index = {}
        for pos, term in enumerate(doc.search_text):
            try:
                current_doc_index[term][1].append(pos)
            except KeyError:
                current_doc_index[term] = [doc.pid, array('I', [pos])]

        # Compute normalization factor for TF
        norm = 0.0
        for _, posting in current_doc_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # Compute TF and DF for each term
        for term, posting in current_doc_index.items():
            freq = len(posting[1])
            tf[term].append(np.round(freq / norm, 4))
            df[term] += 1

        # Merge this document’s index into the global index
        for term, posting in current_doc_index.items():
            index[term].append(posting)

    # Compute IDF
    for term in df:
        idf[term] = np.round(np.log(float(num_documents) / df[term]), 4)

    return index, tf, df, idf, title_index


def rank_documents(terms, docs, index, idf, tf):
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue
        query_vector[termIndex] = (query_terms_count[term] / query_norm) * idf[term]
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    return result_docs


def search_tfidf(query, index, tf, idf):
    """
    The output is the list of documents that contain all of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    # Process query
    query_terms = process_query(query)

    # Edge case: no query terms
    if not query_terms:
        return []

    # For each term, get the set of documents containing it
    doc_sets = []

    for term in query_terms:
        if term in index:
            term_docs = {posting[0] for posting in index[term]}
            doc_sets.append(term_docs)
        else:
            # If any term is missing no doc can satisfy AND condition
            return []

    # Take intersection (AND logic)
    docs = set.intersection(*doc_sets) if doc_sets else set()

    ranked_docs = rank_documents(query_terms, docs, index, idf, tf)

    return ranked_docs
