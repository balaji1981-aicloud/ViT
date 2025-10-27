import re
from extractor.diagram_table_matcher import match_diagram_table_chunks
from Embedder.text_embedder import embed_table_chunks
from vectorstore.chroma_vectorstore import clear_chroma_store, store_embeddings_in_chroma_and_qdrant
from retriever.qdrant_retrieval import build_hybrid_retriever, setup_qdrant_collection, insert_documents
from reranker.bge_reranker import BGEReranker
from llm.query_llm import answer_query_with_context
from dynamic_k import build_vocabulary, extract_terms_from_query, filter_query_with_vocab, extract_terms_fallback
from rank_bm25 import BM25Okapi

# === Paths ===
DIAGRAM_DIR = r"C:\Users\Y8664226\Downloads\pdf_par\output\cropped_pages"
TABLE_DIR = r"C:\Users\Y8664226\Downloads\pdf_par\output\tables"
csv_folder = r"C:\Users\Y8664226\Downloads\pdf_par\output\tables"
CHROMA_DB_DIR = "chroma_store"

def extract_summary(response_text: str) -> str:
    # Capture everything after **Summary** until the next **Section** or end of text
    pattern = r"\*\*Summary\*\*\s*(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return ""

import re

def extract_related_sources(response_text: str):
    # Regex to capture everything after **Related Sources**
    pattern = r"\*\*Related Sources\*\*\s*(?:-.*(?:\n|$))+"
    match = re.search(pattern, response_text, re.IGNORECASE)
    
    if not match:
        return []

    block = match.group(0)

    # Extract each line starting with "-"
    sources = [line.lstrip("-").strip() 
               for line in block.splitlines() if line.strip().startswith("-")]
    
    return sources

# === Initialization ===
def initialize_pipeline():

    vocabulary, voc_set = build_vocabulary(csv_folder)
    print(" Matching diagram and table chunks...")
    chunks = match_diagram_table_chunks(DIAGRAM_DIR, TABLE_DIR)
    print(len(chunks))
    print(f"[DEBUG] First chunk: {chunks[0]}")
    print(" Embedding table rows...")
    documents, embedding_model = embed_table_chunks(chunks)
    print(f"[DEBUG] First chunk: {documents[0]}")
    print(f"len of documents :{len(documents)}")
    print(" Clearing & storing embeddings in ChromaDB...")
    clear_chroma_store(CHROMA_DB_DIR)
    store_embeddings_in_chroma_and_qdrant(documents, embedding_model, CHROMA_DB_DIR)

    client = setup_qdrant_collection()
    dense_model, sparse_model, late_model = insert_documents(client, documents)

    retriever = build_hybrid_retriever(client, dense_model, sparse_model, late_model)
    print("Initializing reranker...")
    reranker = BGEReranker()

    return reranker, retriever, vocabulary, voc_set

# === Main Pipeline ===
def test_pipeline(diagram_dir, table_dir):
    reranker, retriever, vocabulary , voc_set= initialize_pipeline()

    while True:
        query = input("\n Ask a question (or type 'exit'): ")
        top_k = int(input("\n enter k:"))
        if query.lower().strip() == "exit":
            break

        clean_query, _ = filter_query_with_vocab(query, voc_set)
        query_terms = extract_terms_from_query(clean_query, vocabulary)
        len_k = len(query_terms)
        if len_k == 0:
            query_fallback_term = extract_terms_fallback(clean_query, vocabulary)
            len_k = len(query_fallback_term)
                
        if top_k >= len_k: 
            lenr_k , lenrr_k = top_k

        else:
            # final_len = int((len_k - top_k) * 0.7)  
            lenr_k = len_k
            lenrr_k = top_k

        print(f"k is {lenrr_k}")
        print(f"k is {lenr_k}")

        print(" Retrieving documents...")
        results = retriever(query, top_k=lenr_k)
        for i, doc in enumerate(results):
            print(f"{doc.metadata.get('Assembly_name', 'N/A')}") 
    
        print(" Reranking top results...")
        reranked_docs, query_r = reranker.rerank(query, results, top_k=lenrr_k)
        print(f" the query used is {query_r}")
        print(f" {len(reranked_docs)} documents after reranking")
        for i, doc in enumerate(reranked_docs):
            print(f"{doc.metadata.get('Assembly_name', 'N/A')}") 
        
        print("Generating answer...")
        answer = answer_query_with_context(query, reranked_docs)
        response = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        print("\n Retrieved Answer:\n", response)
        summary_text = extract_summary(response)
        matched = extract_related_sources(response)
        print("\n sum:\n", summary_text)
        print("\n matched:\n", matched)

if __name__ == "__main__":
    test_pipeline(DIAGRAM_DIR, TABLE_DIR)




































# import os
# import re
# import numpy as np
# import pandas as pd
# from io import StringIO
# from langchain_core.documents import Document


# from sklearn.metrics.pairwise import cosine_similarity
# from extractor.diagram_table_matcher import match_diagram_table_chunks
# from Embedder.text_embedder import embed_table_chunks
# from Embedder.image_embedder import BlipImageEmbedder
# from vectorstore.chroma_vectorstore import clear_chroma_store, store_embeddings_in_chroma_and_qdrant
# from retriever.hybrid_retriever import build_hybrid_retriever
# from reranker.bge_reranker import BGEReranker
# # from llm.prompt_llm import answer_query_with_context
# from llm.prompt_llm import answer_query_with_context
# from test2 import q_search

# # === Paths ===
# DIAGRAM_DIR = r"C:\Users\Y8664226\Downloads\pdf_par\output\cropped_pages"
# TABLE_DIR = r"C:\Users\Y8664226\Downloads\pdf_par\output\table_updated"
# CHROMA_DB_DIR = "chroma_store"

# # === MMR Filtering ===
# def apply_mmr(query_emb, docs, embeddings, top_k=10, lambda_mult=0.6):
#     if not docs or len(docs) <= top_k:
#         return docs

#     selected = []
#     selected_indices = []
#     query_emb = query_emb.reshape(1, -1)
#     doc_embs = np.array(embeddings)
#     similarities = cosine_similarity(query_emb, doc_embs)[0]

#     for _ in range(top_k):
#         mmr_scores = []
#         for i in range(len(docs)):
#             if i in selected_indices:
#                 continue
#             sim_to_query = similarities[i]
#             sim_to_selected = max(
#                 cosine_similarity(doc_embs[i].reshape(1, -1), doc_embs[selected_indices])[0]
#             ) if selected_indices else 0
#             mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
#             mmr_scores.append((i, mmr_score))
#         selected_idx = max(mmr_scores, key=lambda x: x[1])[0]
#         selected_indices.append(selected_idx)
#         selected.append(docs[selected_idx])

#     return selected

# def filter_docs_by_third_row(query, docs):
    # query_terms = set(re.findall(r"\w+", query.lower()))
#     filtered = []

#     print("\n Filtering by description match in 3rd row, 3rd column...")
#     print(f" Query Terms: {query_terms}\n")

#     for idx, doc in enumerate(docs):
#         print(f"\n Document {idx + 1}: {doc.metadata.get('Assembly_name', 'Unknown Source')}")
        
#         raw_table = doc.metadata.get("table_row", None)
#         if not isinstance(raw_table, str):
#             print(" Skipping: table_row metadata is not a valid string.")
#             continue

#         print("Raw Table String:")
#         print(raw_table)

#         try:
#             df = pd.read_csv(StringIO(raw_table), header=None)
#             print("\n Parsed Table DataFrame:")
#             print(df)
#         except Exception as e:
#             print(f" Failed to parse table from doc: {e}")
#             continue
#         if df.shape[0] < 1 or df.shape[1] < 3:

#             print(" Table does not have enough rows/columns (requires at least 3x3).")
#             continue

#         cell_value = str(df.iloc[0, 2]).lower()
#         print(f"\n Cell value (3rd row, 3rd column): {cell_value}")

#         cleaned_cell = re.sub(r"[^\w\s]", " ", cell_value)
#         cell_tokens = set(cleaned_cell.split())
#         print(f" Cell Tokens: {cell_tokens}")

#         matched_terms = query_terms & cell_tokens
#         print(f" Matched Terms: {matched_terms}")

#         if matched_terms:
#             filtered.append(doc)
#         else:
#             print(" No match found for this doc.")

#     print(f"\n {len(filtered)} documents passed custom filter")
#     return filtered


# # === Initialization ===
# def initialize_pipeline():
#     print(" Matching diagram and table chunks...")
#     chunks = match_diagram_table_chunks(DIAGRAM_DIR, TABLE_DIR)

#     print(" Embedding table rows...")
#     documents, embedding_model = embed_table_chunks(chunks)

#     print(" Clearing & storing embeddings in ChromaDB...")
#     clear_chroma_store(CHROMA_DB_DIR)
#     store_embeddings_in_chroma_and_qdrant(documents, embedding_model, CHROMA_DB_DIR)

#     print(" Building hybrid retriever (BM25 + Dense)...")
#     retriever, dense_retriever, sparse_retriever, vectordb = build_hybrid_retriever(documents, embedding_model, CHROMA_DB_DIR)

#     print("Initializing reranker...")
#     reranker = BGEReranker()

#     return embedding_model, reranker, dense_retriever, sparse_retriever

# def normalize_text(text):
#     # Lowercase, collapse spaces, strip
#     return re.sub(r"\s+", " ", text).strip().lower()

# # === Main Pipeline ===
# def test_pipeline(diagram_dir, table_dir):
#     embedding_model, reranker, dense_retriever, sparse_retriever = initialize_pipeline()

#     while True:
#         query = input("\n Ask a question (or type 'exit'): ")
#         if query.lower().strip() == "exit":
#             break

#         q_retrieval = q_search(query)

#         print(" Retrieving documents...")
#         dense_results = dense_retriever.get_relevant_documents(query)
#         sparse_results = sparse_retriever.get_relevant_documents(query)

#         # === Apply MMR to dense results ===
#         query_emb = embedding_model.embed_query(query)
#         dense_embeddings = [embedding_model.embed_query(doc.page_content) for doc in dense_results]
#         dense_mmr_results = apply_mmr(query_emb, dense_results, dense_embeddings, top_k=10, lambda_mult=0.8)

#         # === Combine with top sparse ===
#         sparse_topk = sparse_results[:10]

#                 # From Chroma
#         dense_mmr_results = [
#             Document(page_content=doc.page_content, metadata={**doc.metadata, "source": "chroma"})
#             for doc in dense_mmr_results
#         ]

#         # From BM25 or sparse retriever
#         sparse_topk = [
#             Document(page_content=doc.page_content, metadata={**doc.metadata, "source": "bm25"})
#             for doc in sparse_topk
#         ]

#         # === Deduplicate ===
#         combined = sparse_topk + q_retrieval
#         priority_order = {"qdrant": 1, "bm25": 2}  # smaller = higher priority

#         unique = {}
#         for doc in combined:
#             norm_text = normalize_text(doc.page_content)
#             asm_name = doc.metadata.get("Assembly_name", "").strip().lower()
#             key = (norm_text, asm_name)

#             source = doc.metadata.get("source", "").lower()
#             if key not in unique or priority_order.get(source, 99) < priority_order.get(unique[key].metadata.get("source", "").lower(), 99):
#                 unique[key] = doc

#         deduped_results = list(unique.values())

#         print(f"{len(deduped_results)} documents after MMR + deduplication")
#         print(" Showing top documents:")
#         for i, doc in enumerate(deduped_results):
#             print(f"{doc.metadata.get('table_row', 'N/A')}") 


#         print(" Reranking top results...")
#         reranked_docs = reranker.rerank(query, deduped_results, top_k=10)
#         print(f" {len(reranked_docs)} documents after reranking")
#         for i, doc in enumerate(reranked_docs):
#             print(f"{doc.metadata.get('table_row', 'N/A')}") 


#         print(f"Filtering by description match in 3rd row, 3rd column...")
#         filtered_docs = filter_docs_by_third_row(query, reranked_docs)
#         print(f"{len(filtered_docs)} documents passed custom filter")

#         if not filtered_docs:
#             print(" No documents matched the query in the description field.")
#             continue
        
        
#         print("Generating answer...")
#         answer = answer_query_with_context(query, reranked_docs)
#         response = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

#         print("\n Retrieved Answer:\n", response)

# if __name__ == "__main__":
#     test_pipeline(DIAGRAM_DIR, TABLE_DIR)



