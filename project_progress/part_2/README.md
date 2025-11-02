# Part 2: Indexing and Evaluation

This part of the project focuses on building the actual search engine and checking how well it works. It includes the creation of the TF-IDF index and the evaluation of its performance using different metrics.

## What I Did

1. Built an **Inverted Index** that supports **TF-IDF ranking**.  
2. Implemented **Conjunctive (AND)** search for multi-term queries.  
3. Added a full **Evaluation module** with seven metrics:  
   - Precision@K (P@K)  
   - Recall@K (R@K)  
   - F1-Score@K (F1@K)  
   - Average Precision@K (AP@K)  
   - Mean Average Precision (MAP)  
   - Mean Reciprocal Rank (MRR)  
   - Normalized Discounted Cumulative Gain (NDCG)  

---

## File Structure

| File | Description |
|------|--------------|
| **`run_search.py`** | Run this first. It processes all products, builds the TF-IDF index, and saves it as `irwa_index.pkl`. It also runs 5 test queries and saves their results in `search_results.txt`. |
| **`evaluation_query.py`** | Used for the 2 predefined queries (from the professor). It loads the index and evaluates results using `data/validation_labels.csv`. |
| **`my_query_evaluation.py`** | Used for my 5 custom queries. It loads the same index and evaluates results using my own `data/my_queries_validation_labels.csv`. |
| **`indexing.py`** | Contains the functions to create and search the TF-IDF index. |
| **`evaluation.py`** | Contains all metric functions (P@K, MAP, NDCG, etc.). |
| **`query_preparation.py`** | Cleans and processes queries (tokenization, stopwords, stemming). |
| **`irwa_index.pkl`** | Saved index file. |
| **`search_results.txt`** | File that shows the top results for my custom queries. |

---

## How to Run Everything

All commands should be run from the root folder of the project:
`irwa-search-engine/`

---

### Step 1: Build the Index

This step reads and processes the full dataset, builds the TF-IDF index, and saves it.  
You only need to do this once.

```bash
python project_progress/part_2/run_search.py
