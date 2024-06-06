import torch
from sentence_transformers.util import semantic_search


def get_document_texts(dataset, qrels, all_docs=True):
    document_ids = []
    document_texts = []
    for doc in dataset.get_corpus_iter():
        if not all_docs and doc["docno"] not in qrels["docno"].values:
            continue
        document_ids.append(doc["docno"])
        document_texts.append(doc["text"])
    torch.save(document_ids, "document_ids.pt")
    torch.save(document_texts, "document_texts.pt")
    return document_ids, document_texts


def get_query_list(queries):
    query_list = []
    for i in range(len(queries)):
        query_list.append(queries["query"].iloc[i])
    return query_list


def calculate_embeddings(embedder, document_texts, query_list):
    document_embeddings = embedder.encode(document_texts, show_progress_bar=True, convert_to_tensor=True)
    query_embeddings = embedder.encode(query_list, show_progress_bar=True, convert_to_tensor=True)
    torch.save(document_embeddings, "document_embeddings_mini_lm.pt")
    torch.save(query_embeddings, "query_embeddings_mini_lm.pt")
    return document_embeddings, query_embeddings


def get_k_documents(query, most_relevant_docs, embedder, document_embeddings, document_texts, k=10, threshold=0.5):
    documents = []
    query_embed = embedder.encode(query, convert_to_tensor=True)
    if most_relevant_docs:
        relevant_embed = torch.zeros_like(query_embed)
        for doc in most_relevant_docs:
            relevant_embed += embedder.encode(doc, convert_to_tensor=True)
        relevant_embed /= len(most_relevant_docs)
        query_embed = 0.7 * query_embed + 0.3 * relevant_embed
    top_k = semantic_search(query_embed, document_embeddings, top_k=k)[0]
    i = 1
    for hit in top_k:
        dct = {}
        # add to list if the score is higher than the threshold
        if hit['score'] > threshold:
            dct['title'] = f"Answer {i}"
            dct['answer'] = document_texts[hit['corpus_id']]
            dct['score'] = hit['score'] * 100
            documents.append(dct)
            i += 1
    return documents


def print_retrieved_documents(documents):
    for i, doc in enumerate(documents):
        print(f"Answer {i + 1}: {doc}")
