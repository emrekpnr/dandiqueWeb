import argparse
import pyterrier as pt
import torch
from sentence_transformers import SentenceTransformer

from .utils import get_k_documents, get_document_texts, get_query_list, calculate_embeddings, print_retrieved_documents

import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    if not pt.started():
        pt.init()

    dataset = pt.get_dataset('irds:antique/train/split200-train')
    test_dataset = pt.get_dataset('irds:antique/test')

    queries = test_dataset.get_topics()
    qrels = test_dataset.get_qrels()

    embedder = SentenceTransformer("msmarco-MiniLM-L-6-v3").to(device)
    # embedder = SentenceTransformer("msmarco-distilbert-base-v4").to(device)

    print("Loading documents from file...")
    try:
        document_ids = torch.load("document_ids.pt")
        document_texts = torch.load("document_texts.pt")
    except FileNotFoundError:
        print("Documents not found. Getting documents...")
        document_ids, document_texts = get_document_texts(dataset, qrels)

    query_list = get_query_list(queries)

    print("Loading embeddings from file...")
    try:
        document_embeddings = torch.load("document_embeddings_mini_lm.pt")
        query_embeddings = torch.load("query_embeddings_mini_lm.pt")
    except FileNotFoundError:
        print("Embeddings not found. Embedding documents and queries...")
        document_embeddings, query_embeddings = calculate_embeddings(embedder, document_texts, query_list)

    print("Done!")
    return embedder, document_embeddings, document_texts


def search_query(query, embedder, document_embeddings, document_texts, threshold=0.5):
    documents = get_k_documents(query, None, embedder, document_embeddings, document_texts, k=10, threshold=threshold)
    # sort them with new metric which is length of answer * score
    documents = sorted(documents, key=lambda x: len(x['answer']) * x['score'], reverse=True)
    # also filter out documents with length less than 3 words
    documents = [doc for doc in documents if len(doc['answer'].split()) > 10 * 2 * threshold]
    for i, doc in enumerate(documents):
        doc['title'] = f"Answer {i + 1}"
    return documents


def relevance_feedback(search_text, most_relevant_docs, embedder, document_embeddings, document_texts):
    # search_text += " " + first_result
    documents = get_k_documents(search_text, most_relevant_docs, embedder, document_embeddings, document_texts, k=10, threshold=0.5)
    documents = sorted(documents, key=lambda x: 0.5 * len(x['answer'].split()) + x['score'], reverse=True)
    # also filter out documents with length less than 3 words
    documents = [doc for doc in documents if len(doc['answer'].split()) > 10]
    for i, doc in enumerate(documents):
        doc['title'] = f"Answer {i + 1}"
    return documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold similarity score for document retrieval")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = pt.get_dataset('irds:antique/train/split200-train')
    test_dataset = pt.get_dataset('irds:antique/test')

    queries = test_dataset.get_topics()
    qrels = test_dataset.get_qrels()

    # embedder = SentenceTransformer("msmarco-MiniLM-L-6-v3").to(device)
    embedder = SentenceTransformer("msmarco-distilbert-base-v4").to(device)

    print("Loading documents from file...")
    try:
        document_ids = torch.load("document_ids.pt")
        document_texts = torch.load("document_texts.pt")
    except FileNotFoundError:
        print("Documents not found. Embedding documents...")
        document_ids, document_texts = get_document_texts(dataset, qrels)

    query_list = get_query_list(queries)

    print("Loading embeddings from file...")
    try:
        document_embeddings = torch.load("document_embeddings.pt")
        query_embeddings = torch.load("query_embeddings.pt")
    except FileNotFoundError:
        print("Embeddings not found. Embedding documents and queries...")
        document_embeddings, query_embeddings = calculate_embeddings(embedder, document_texts, query_list)

    print("Done!")

    while True:
        try:
            query = input("Enter query: ")
        except EOFError:
            break
        documents = get_k_documents(query, embedder, document_embeddings, document_texts, k=args.k, threshold=args.threshold)
        print_retrieved_documents(documents)
        print("\nPress Ctrl+D to quit")
