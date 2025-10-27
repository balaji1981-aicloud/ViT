import streamlit as st
import os
import csv
import ollama
import json
import uuid
import re
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams,PointStruct
from uuid import uuid4
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import PointStruct, VectorParams,Distance, VectorParamsDiff, SparseVectorParams, Modifier, MultiVectorConfig, MultiVectorComparator, HnswConfigDiff
import uuid
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.models import SparseVector
from langchain_core.documents import Document
from fastembed import LateInteractionTextEmbedding,SparseTextEmbedding
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.retrievers import BM25Retriever
from reranker.bge_reranker import BGEReranker
from langchain.document_loaders import CSVLoader
import tokenizers
from tokenizers import Tokenizer

# Set the path to the folder containing your CSV files
folder_path = r"C:\Users\a5348634\OneDrive - Saint-Gobain\Documents\backup\Life Science\Design Recommender\output\tables_modified"
# Set the name of your output JSON file
output_json_file = 'output.json'
client = QdrantClient(url="http://10.87.60.30:6333")
collection_name = "hybrid-search-Assembly"
dense_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/llm-embedder", #bge-base-en-v1.5
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
sparse_model = SparseTextEmbedding("Qdrant/bm25")
late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
inputcsv="Complete_Assembly.csv"

def calculate_tokencount(prompt):
    tokenizer = Tokenizer.from_pretrained("TheBloke/Llama-2-70B-fp16")
    encoding_result=tokenizer.encode(prompt)
    print("Token length:",encoding_result)

# ========== Step 1: Load Excel File ==========
def load_csv_to_chunks(file_path):
    # Load the documents
    loader = CSVLoader(file_path)
    documents=loader.load()
    #print(documents)
    return documents

def convert_csv_to_json():
    # Dictionary to hold the combined data
    all_csv_data = {}

    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            filepath = os.path.join(folder_path, filename)
            
            # List to store the data from the current CSV file
            file_data = []
            
            try:
                # Open and read the CSV file
                with open(filepath, mode='r', encoding='utf-8') as csv_file:
                    # Use DictReader to automatically handle headers as dictionary keys
                    csv_reader = csv.DictReader(csv_file)
                    
                    # Append each row (as a dictionary) to the file_data list
                    for row in csv_reader:
                        file_data.append(row)
                
                # Add the filename as a key and its data as the value to the master dictionary
                all_csv_data[str(filename).split('_')[0]] = file_data
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Write the combined dictionary to a single JSON file
    if all_csv_data:
        try:
            with open(output_json_file, mode='w', encoding='utf-8') as json_file:
                # Use json.dump for a human-readable format
                json.dump(all_csv_data, json_file, indent=4)
            print(f"Successfully converted CSVs to a single JSON file: {output_json_file}")
        except Exception as e:
            print(f"Error writing to JSON file: {e}")
    else:
        print("No CSV files found or processed.")


def convert_json_to_chunks():
    # ---------- Step 1: Load JSON ----------
    with open(r"output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for key, value in data.items():
        # Convert value (list/dict) to text
        value_text = json.dumps(value, indent=2, ensure_ascii=False)
        chunk_text = f"{key}: {value_text}"
        
        chunks.append({
            "id": str(uuid.uuid4()),
            "assembly": key,
            "text": chunk_text,
            "raw_value": value
        })
    print(f"✅ Created {len(chunks)} top-level chunks.")
    return chunks

def convert_chunks_to_docs(chunks):
    docs = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk["text"],   # main text for embedding
                metadata={
                    "assembly": chunk["assembly"],
                    "id": chunk["id"]
                }
            )
        )

    print(f"✅ Prepared {len(docs)} documents.")
    return docs

def load_bom_csvs_as_documents(folder_path, rows_per_chunk=5):
    documents = []

    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            assembly_name = os.path.splitext(filename)[0].split('_')[0]

            # Read the BOM data
            df = pd.read_csv(file_path)

            # Iterate through the dataframe in chunks of 'rows_per_chunk'
            for start in range(0, len(df), rows_per_chunk):
                end = min(start + rows_per_chunk, len(df))
                chunk_df = df.iloc[start:end]

                # Build readable text for this chunk
                chunk_texts = []
                for _, row in chunk_df.iterrows():
                    row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                    chunk_texts.append(f"---\n{row_text}")

                # Combine into one chunk document
                chunk_content = (
                    f"Assembly Name: {assembly_name}\n\n"
                    f"Components (Items {start+1}–{end}):\n"
                    + "\n\n".join(chunk_texts)
                )

                # Create LangChain Document
                doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "assembly_name": assembly_name,
                        "source_file": filename,
                        "chunk_start_row": start + 1,
                        "chunk_end_row": end,
                        "total_rows": len(df),
                    },
                )

                documents.append(doc)

    return documents

def create_qdrant_collection():
    # Delete if exists
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted old collection '{collection_name}'")
    except:
        pass
    # Create with dense, sparse, and late-interaction slots
    client.recreate_collection(
            collection_name="hybrid-search-Assembly",
            vectors_config={
                "llmembedder": VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
                "colbertv2.0": VectorParams(
                    size=128,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=HnswConfigDiff(m=0),  # disable HNSW for reranking
                ),
                },
            sparse_vectors_config={
                "bm25": SparseVectorParams(modifier=Modifier.IDF)
            }
        )
    print(f"Created collection '{collection_name}'")

def upload_embeddings_to_collection(docs):
    points = []
    for idx, doc in enumerate(docs):
        text = doc.page_content if isinstance(doc, Document) else str(doc)

        dense_vec = dense_model.embed_query(text)
        sparse_obj = next(sparse_model.embed(text))
        late_vec = next(late_model.embed(text))

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                    "llmembedder": dense_vec,
                    "bm25": sparse_obj.as_object(),
                    "colbertv2.0": late_vec,
                },
            payload={"text": doc.page_content, **doc.metadata} 

            )
        #print(point)
        points.append(point)

    #client.upsert(
     #       collection_name='hybrid-search-Assembly',
      #      points=points
      #  )

    #print(point)
    batch_size=100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name='hybrid-search-Assembly',
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}")
        
    print(f"Inserted {len(points)} points with dense, sparse, and ColBERT vectors.")

# Delete collection
def delete_collection():
    # Delete the collection
    client.delete_collection(collection_name='hybrid-search-Assembly')

def query_qdrant(query,top_k):

    PROMPT_TEMPLATE = """
        The context shows the details of a single use life science assembly:Assembly name and Component details like item number,the quantity or number of components,component description,material the component is made of ,component number and component details like dimension,ID,OD,capacity.
        An assembly may contain multiple components.
        When checking for a component name strictly check both in Description value and Dimension value.
        Answer the question based only on the following context:

       {context}

        ---

       Answer the question based on the above context: {question}
      """
    #embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embedding=DefaultEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Search the DB.
    # Make a retriever from the vector store
    #results = vectordb.similarity_search(query, k=5)
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    dense_query = dense_model.embed_query(query)
    sparse_query_obj = next(sparse_model.embed(query))
    late_query = next(late_model.embed(query))
    prefetch = [
            models.Prefetch(
                query=dense_query,
                using="llmembedder",
                limit=top_k,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_obj.as_object()),
                using="bm25",
                limit=top_k,
            ),
        ]
    results = client.query_points(
            'hybrid-search-Assembly',
            prefetch=prefetch,
            query=late_query,
            using="colbertv2.0",
            with_payload=True,
            limit=top_k,
        )
    
    #print(results)
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #Using BM25 Retriever
    #bm25_retriever = BM25Retriever.from_documents(results)
    # Set top_n if you want fewer results after BM25 re-ranking
    ##bm25_retriever.k = 7  # Number of final results after BM25
    # Run BM25 retriever on the query
    ##final_docs = bm25_retriever.invoke(query)
    ##print(final_docs)
    # Convert Qdrant points to LangChain Document 
    documents = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.pop("text", "")
        docs = Document(page_content=text, metadata=payload)
        documents.append(docs)
    reranker = BGEReranker()
    reranked_docs, query_r = reranker.rerank(query, documents, top_k=5)
    print(f" the query used is {query_r}")
    print(f" {len(reranked_docs)} documents after reranking")
    for i, doc in enumerate(reranked_docs):
        print(f"{doc.metadata.get('assembly_name', 'N/A')}") 
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    print("After Retrieval context text:")
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    calculate_tokencount(prompt)
    model = Ollama(model="qwen3:4b",temperature=0)
    response_text = model.invoke(prompt)
    return response_text


# -----------------------
# LLM + DB Setup
# -----------------------
#llm = OllamaLLM(model="qwen3:latest", temperature=0)

def remove_think_tags(text_with_think):
    """
    Removes content within <think></think> tags from a string.

    Args:
        text_with_think: The input string potentially containing <think> tags.

    Returns:
        The string with <think> tags and their contents removed.
    """
    # The regex /<think>.*?<\/think>/gs matches:
    # <think>: The literal opening tag.
    # .*: Any character (except newline by default).
    # ?: Makes the preceding * non-greedy, matching the shortest possible string.
    # <\/think>: The literal closing tag (escaped /).
    # The 're.DOTALL' flag allows '.' to match newline characters.
    # The 're.IGNORECASE' flag makes the matching case-insensitive.
    cleaned_text = re.sub(r"<think>.*?</think>", "", text_with_think, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text


#convert_csv_to_json()
#chunks=convert_json_to_chunks()
#docs=convert_chunks_to_docs(chunks)
#docs=load_csv_to_chunks(inputcsv)
#path=r"C:\Users\a5348634\OneDrive - Saint-Gobain\Documents\backup\Life Science\Design Recommender\outputold\tables_modified"
#docs=load_bom_csvs_as_documents(path)
#create_qdrant_collection()
#upload_embeddings_to_collection(docs)
# -----------------------
# Streamlit Chat UI
# -----------------------
st.title("💬 DREAM Chatbot")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
if user_query := st.chat_input("Ask about your assemblies, components, or dimensions..."):
    # Display user message
    st.chat_message("user").write(user_query)
    response=query_qdrant(user_query,15)
    # Show assistant message
    with st.chat_message("assistant"):
            print("LLM Answer:", response)
            result=remove_think_tags(response)
            st.write(result)
            #st.markdown("**Results:**")
            #st.write(results)

    # Save interaction in history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content":result })
