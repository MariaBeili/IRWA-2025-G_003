import random
import numpy as np
import pickle
import os
import time
from project_progress.part_2.query_preparation import process_query
from project_progress.part_3.ranking import rank_tfidf_cosine, rank_bm25, load_index

from myapp.search.objects import Document
from project_progress.part_1.data_preparation import ProcessedDocument
from project_progress.part_2.indexing import create_index_tfidf


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method that returns random <num_results> documents from the corpus.
    Useful for testing the UI without a working backend.
    
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    # Get all document IDs
    doc_ids = list(corpus.keys())
    # Pick random IDs
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        # Create a document object with a fake ranking score
        res.append(Document(
            pid=doc.pid, 
            title=doc.title, 
            description=doc.description,
            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), 
            ranking=random.random()
        ))
    return res


class SearchEngine:
    """
    Class that implements the search logic.
    It loads the necessary indices and models at startup and provides a search method.
    """
    
    def __init__(self, corpus=None):
        """
        Initializes the Search Engine by loading:
        1. The TF-IDF inverted index (built in Part 2).
        2. The BM25 and custom model (optional, for Part 3 advanced ranking).
        """
        print("Initializing Search Engine...")


        if corpus == None:
            print("Loading Presaved Index...")
            # 1. Load prebuilt TF-IDF / BM25 indices from the pickle file
            # ensure 'project_progress/part_2/irwa_index.pkl' exists
            try:
                self.index, self.tf, self.idf, self.title_index = load_index()
                print("TF-IDF Index loaded successfully.")
            except Exception as e:
                print(f"Error loading TF-IDF index: {e}")
                self.index, self.tf, self.idf = {}, {}, {}
        
        else:
            print("Calculating Index from Corpus...")
            
            # Example documents
            print("Processing documents...")
            start_time = time.time()

            processed_corpus = [None] * len(corpus)

            for i in range(len(corpus)):
                processed_corpus[i] = ProcessedDocument.from_document(list(corpus.values())[i])
                processed_corpus[i].process_fields()

            process_time = time.time() - start_time
            print(f"Documents processed in {process_time:.4f} seconds.\n")


            # Time the index creation
            print("Building inverted index and computing TF-IDF...")
            start_time = time.time()

            self.index, self.tf, df, self.idf, self.title_index = create_index_tfidf(processed_corpus)

            index_time = time.time() - start_time
            print(f"Index built in {index_time:.4f} seconds.\n")
        

    def search(self, query, corpus=None, method="tfidf", topN=20):
        """
        Main search function.
        
        :param query: The user's search string (e.g., "red shoes").
        :param search_id: Unique ID for analytics (optional).
        :param corpus: The full dataset of documents (dictionary mapping PID -> Document).
        :param method: The ranking method to use ('tfidf', 'bm25', 'word2vec', 'custom').
        :return: A list of relevant documents (dictionaries) ready for the UI.
        """
        
        # 1. Preprocess the query (stemming, stopword removal)
        query_terms = process_query(query)

        # 2. Choose the Ranking Method
        results = []
        
        if method == "tfidf":
            # Part 2 standard ranking
            results = rank_tfidf_cosine(query_terms, self.index, self.tf, self.idf)
            
        elif method == "bm25":
            # Part 3 BM25 ranking
            # Check if your rank_bm25 needs 'tf' or just 'idf' based on your implementation
            # Assuming it takes (query, index, idf) or similar. Adjust arguments if needed.

            try:
                results = rank_bm25(query_terms, self.index, self.tf, self.idf)
            except Exception as e:
                print(f"Error in BM25: {e}. Falling back to TF-IDF.")
                results = rank_tfidf_cosine(query_terms, self.index, self.tf, self.idf)

                
        elif method == "custom":
            # Part 3 Custom Score (e.g., TF-IDF boosted by rating)
            # First get TF-IDF scores
            base_results = rank_tfidf_cosine(query_terms, self.index, self.tf, self.idf)
            
            # Apply boosting logic
            custom_results = []
            for pid, score in base_results:
                doc = corpus.get(pid)
                if doc and hasattr(doc, 'average_rating') and doc.average_rating:
                    # Example formula: score * (1 + rating/10)
                    # A 5-star product gets a 50% boost
                    try:
                        rating = float(doc.average_rating)
                        boost = 1 + (rating / 10.0)
                        new_score = score * boost
                    except ValueError:
                        new_score = score
                    custom_results.append((pid, new_score))
                else:
                    custom_results.append((pid, score))
            
            # Re-sort after boosting
            custom_results.sort(key=lambda x: x[1], reverse=True)
            results = custom_results
            
        else:
            # Default fallback
            results = rank_tfidf_cosine(query_terms, self.index, self.tf, self.idf)

        pids = []

        for pid, score in results[:topN]:
            pids.append(pid)

        return pids
