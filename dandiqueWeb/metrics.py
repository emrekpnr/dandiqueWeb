import numpy as np


def mean_average_precision(query_id, ranked_documents, qrels):
    relevant_docs = qrels[qrels["qid"] == query_id]
    relevant_docs = relevant_docs[relevant_docs["label"] != 1]
    num_relevant_docs = len(relevant_docs)
    average_precision = 0
    num_retrieved_relevant_docs = 0
    for i, doc in enumerate(ranked_documents):
        if doc in relevant_docs["docno"].values:
            num_retrieved_relevant_docs += 1
            average_precision += num_retrieved_relevant_docs / (i + 1)
    return average_precision / num_relevant_docs


def mean_reciprocal_rank(query_id, ranked_documents, qrels):
    relevant_docs = qrels[qrels["qid"] == query_id]
    relevant_docs = relevant_docs[relevant_docs["label"] != 1]
    for i, doc in enumerate(ranked_documents):
        if doc in relevant_docs["docno"].values:
            return 1 / (i + 1)
    return 0


def precision_at_k(query_id, ranked_documents, qrels, k):
    relevant_docs = qrels[qrels["qid"] == query_id]
    relevant_docs = relevant_docs[relevant_docs["label"] != 1]
    num_retrieved_relevant_docs = 0
    for i, doc in enumerate(ranked_documents[:k]):
        if doc in relevant_docs["docno"].values:
            num_retrieved_relevant_docs += 1
    return num_retrieved_relevant_docs / k


def ndcg_at_k(query_id, ranked_documents, qrels, k):
    relevant_docs = qrels[qrels["qid"] == query_id]
    relevant_docs = relevant_docs[relevant_docs["label"] != 1]
    dcg = 0
    idcg = 0
    for i, doc in enumerate(ranked_documents[:k]):
        if doc in relevant_docs["docno"].values:
            dcg += relevant_docs[relevant_docs["docno"] == doc]["label"].values[0] / np.log2(i + 2)
        idcg += 4 / (np.log2(i + 2))
    return dcg / idcg
