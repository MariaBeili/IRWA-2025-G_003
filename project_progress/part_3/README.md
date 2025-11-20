# Part 3: Ranking & Filtering 
This part we focus on ranking methods, how we tested them with the queries from Part2, how to run the code and where all the outputs are stired.

## What this part of the project does
The goal of Part 3 is to experiment with different ranking approaches in a search engine. All queries follow *AND semantics*, so a document is only retrieved if ti contains all the query terms. 

For this part, we implemented:
1. TF-IDF + Cosine Similarity 
2. BM25
3. Our own Score Ranking: "50% TF-IDF + Cosine Similarity, 50% price-rating Cosine Similarity"
4. Word2vec + cosine

Then we tested these methos using the 5 queries, so we could compare how each methos chaneges the ranking order. 

-----

## File Structure

| File | Description |
|------|--------------|
| **`ranking.ipynb`** | Main script with ranking methods. |
| **`ranking_results.txt `** | Output file with all results for my 5 queries  . |


and we reuse the index built in Part 2 `project_progress/part_2/irwa_index.pkl`

---------


## How to run 

### Install the gensim module
First of all, you will have to run the following command to be able to read the word embeddings.
```bash
pip install gensim
```

### Download the word embeddings
Then, download the word2vec model from this link and store it in the resources folder.
https://www.kaggle.com/datasets/suraj520/googlenews-vectors-negative300bingz-gz-format?resource=download

### Run the ranking script
```bash
python project_progress/part_3/ranking.py
```
Outputs:
* print the Top-5 results in the terminal,
* save all results into ranking_results.txt.



