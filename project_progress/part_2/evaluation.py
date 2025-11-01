import math


def compute_precision_at_K(docs, benchmark, K):
    relevant_docs_retrieved = 0

    if K == 0 or len(docs) == 0:
        return 0.0
    elif K > len(docs):
        K = len(docs)

    for doc in docs[:K]:
        if doc in benchmark:
            relevant_docs_retrieved += 1

    return relevant_docs_retrieved / K

def compute_recall_at_K(docs, benchmark, K):
    relevant_docs_retrieved = 0

    if K == 0 or len(benchmark) == 0:
        return 0.0
    elif K > len(benchmark):
        K = len(benchmark)

    for doc in benchmark[:K]:
        if doc in docs:
            relevant_docs_retrieved += 1

    return relevant_docs_retrieved / K

def compute_average_precision_at_K(docs, benchmark, K):
    true_positives_seen = 0
    index = 0
    average_precision = 0

    for doc in docs[:K]:
        index += 1
        if doc in benchmark:
            true_positives_seen+=1
            precision = true_positives_seen / index
            average_precision += precision

    if true_positives_seen == 0:
        return 0.0
    return average_precision / true_positives_seen

def compute_F1_score_at_K(docs, benchmark, K):
    recall = compute_recall_at_K(docs, benchmark, K)
    precision = compute_precision_at_K(docs, benchmark, K)
    if recall == 0 and precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)

def compute_mean_average_precision(rankings, benchmarks):
    total = 0
    for ranking, rel in zip(rankings, benchmarks):
        total += compute_average_precision_at_K(ranking, rel, len(ranking))
    return total / len(rankings)

def compute_mean_reciprocal_rank(rankings, benchmarks):
    total = 0
    for ranking, rel in zip(rankings, benchmarks):
        for idx, doc in enumerate(ranking, start=1):
            if doc in rel:
                total += 1 / idx
                break
    return total / len(rankings)

def compute_normalized_discounted_cumulative_gain(docs, benchmark):
    dcg = 0
    for i, doc in enumerate(docs, start=1):
        if doc in benchmark:
            dcg += 1 / math.log2(i + 1)

    ideal_relevants = len(benchmark)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_relevants + 1))

    return dcg / idcg if idcg > 0 else 0.0

