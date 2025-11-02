# Part 2: Indexing and Evaluation

This part of the project focuses on building the actual search engine and checking how well it works. It includes the creation of the TF-IDF index and the evaluation of its performance using different metrics.

## What We Did

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
| **`run_search.ipynb`** | Run this first. It processes all products, builds the TF-IDF index, and saves it as `irwa_index.pkl`. It also runs 5 test queries and saves their results in `search_results.txt`. |
| **`evaluation_query.ipynb`** | Used for the 2 predefined queries (from the professor). It loads the index and evaluates results using `data/validation_labels.csv`. |
| **`my_query_evaluation.ipynb`** | Used for 5 custom queries. It loads the same index and evaluates results using own `data/my_queries_validation_labels.csv`. |
| **`indexing.ipynb`** | Contains the functions to create and search the TF-IDF index. |
| **`evaluation.py`** | Contains all metric functions (P@K, MAP, NDCG, etc.). |
| **`query_preparation.ipynb`** | Cleans and processes queries (tokenization, stopwords, stemming). |
| **`irwa_index.pkl`** | Saved index file. (too big to upload to github) |
| **`search_results.txt`** | File that shows the top results for custom queries. |
| **`my_queries_validation_labels`** | This file contains the *manual ground truth* for 5 custom queries. We created it after manually checking the top retrieved results and assigning binary labels (1 = relevant, 0 = not relevant). |

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
```
Outputs:
* project_progress/part_2/irwa_index.pkl → the saved index
* project_progress/part_2/search_results.txt → results for 5 test queries

### Step 2: Run Evaluation 1 (Provided Queries)
This part checks the system with the two predefined queries that come in the professor’s file data/validation_labels.csv.
```bash
python project_progress/part_2/evaluation_query.py
```
Output: It prints a table with the metrics (P@K, R@K, F1@K, AP@K, MAP, MRR, NDCG) for both queries.

### Step 3: Run Evaluation 2 (Custom Queries)
This part is for own queries. We created a new ground truth file called data/my_queries_validation_labels.csv, and we used it to evaluate 5 queries.
```bash
python project_progress/part_2/my_query_evaluation.py
```
Output: It prints the metrics for all 5 queries, and also the overall MAP and MRR at the end.



