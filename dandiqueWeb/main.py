import argparse
import pyterrier as pt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, ndcg_at_k
from utils import get_k_documents

import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_embeddings", action="store_true", help="Load embeddings from file")
    parser.add_argument("--load_documents", action="store_true", help="Load documents from file")
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for document retrieval")
    parser.add_argument("--num_rf", type=int, default=1, help="Number of relevant feedback documents")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = pt.get_dataset('irds:antique/train/split200-train')
    test_dataset = pt.get_dataset('irds:antique/test')

    index = pt.IndexFactory.of("./indices/antique")
    print(index.getCollectionStatistics().toString())

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    res = pt.Experiment(
        [bm25],
        test_dataset.get_topics(),
        test_dataset.get_qrels(),
        ["map", "recip_rank", "P_5", "P_10", "ndcg_cut_5", "ndcg_cut_10"],
        names=["BM25"]
    )

    # Neural IR
    queries = test_dataset.get_topics()
    qrels = test_dataset.get_qrels()

    embedder = SentenceTransformer("msmarco-MiniLM-L-6-v3")
    # embedder = SentenceTransformer("msmarco-distilbert-base-v4").to(device)

    if args.load_documents:
        print("Loading documents from file...")
        document_ids = torch.load("document_ids.pt")
        document_texts = torch.load("document_texts.pt")
    else:
        print("Saving documents...")
        document_ids = []
        document_texts = []
        for doc in dataset.get_corpus_iter():
            if doc["docno"] in qrels["docno"].values:
                document_ids.append(doc["docno"])
                document_texts.append(doc["text"])
        torch.save(document_ids, "document_ids.pt")
        torch.save(document_texts, "document_texts.pt")

    query_list = []
    for i in range(len(queries)):
        query_list.append(queries["query"].iloc[i])

    if args.load_embeddings:
        print("Loading embeddings from file...")
        document_embeddings = torch.load("document_embeddings_mini_lm.pt")
        query_embeddings = torch.load("query_embeddings_mini_lm.pt")
    else:
        print("Embedding documents and queries...")
        document_embeddings = embedder.encode(document_texts, show_progress_bar=True, convert_to_tensor=True)
        query_embeddings = embedder.encode(query_list, show_progress_bar=True, convert_to_tensor=True)
        torch.save(document_embeddings, "document_embeddings.pt")
        torch.save(query_embeddings, "query_embeddings.pt")
    print("Done!")

    metrics = pd.DataFrame(columns=["Query", "MAP", "MRR", "P@5", "P@10", "NDCG@5", "NDCG@10"])

    for i, query_embed in enumerate(query_embeddings):
        ranked_docs = []
        # print(f"Query: {queries['query'].iloc[i]}")
        results = get_k_documents(query_list[i], None, embedder, document_embeddings, document_texts, k=10, threshold=0.5)
        most_relevant_docs = []
        for doc in results[:args.num_rf]:
            most_relevant_docs.append(doc['answer'])
        new_results = get_k_documents(query_list[i], most_relevant_docs, embedder, document_embeddings, document_texts, k=100, threshold=0.0)
        for result in new_results:
            docno = document_ids[result['docno']]
            ranked_docs.append(docno)

        map = mean_average_precision(queries["qid"].iloc[i], ranked_docs, qrels)
        mrr = mean_reciprocal_rank(queries["qid"].iloc[i], ranked_docs, qrels)
        p_5 = precision_at_k(queries["qid"].iloc[i], ranked_docs, qrels, 5)
        p_10 = precision_at_k(queries["qid"].iloc[i], ranked_docs, qrels, 10)
        ndcg_5 = ndcg_at_k(queries["qid"].iloc[i], ranked_docs, qrels, 5)
        ndcg_10 = ndcg_at_k(queries["qid"].iloc[i], ranked_docs, qrels, 10)
        metrics = pd.concat([metrics, pd.DataFrame([[queries["query"].iloc[i], map, mrr, p_5, p_10, ndcg_5, ndcg_10]],
                                                   columns=["Query", "MAP", "MRR", "P@5", "P@10", "NDCG@5",
                                                            "NDCG@10"])])

    res = pd.concat([res, pd.DataFrame([["Neural IR", metrics['MAP'].mean(), metrics['MRR'].mean(), metrics['P@5'].mean(),
                                         metrics['P@10'].mean(), metrics['NDCG@5'].mean(), metrics['NDCG@10'].mean()]],
                                        columns=["name", "map", "recip_rank", "P_5", "P_10", "ndcg_cut_5", "ndcg_cut_10"])])

    print(res)
