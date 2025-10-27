import pandas as pd
import glob
import os
import csv
import re

def build_vocabulary(csv_folder):
    vocabulary = []       # list of tokenized descriptions
    vocab_flat = set()    # deduplicated tokens

    for file_name in os.listdir(csv_folder):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(csv_folder, file_name)
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 3:
                continue  # Not enough rows

            headers = rows[1]  # Second row = headers
            if len(headers) < 3:
                continue

            if headers[2].strip().upper() != "DESCRIPTION":
                continue

            # Process DESCRIPTION column
            for row in rows[2:]:
                if len(row) >= 3 and row[2].strip():
                    term = row[2].strip()
                    # Extract alphabetic tokens only
                    token =  re.findall(r"\w+", term.lower())
                    tokens = re.findall(r"[A-Za-z]+", term.lower())
                    
                    if tokens:
                        vocabulary.append(token)
                        vocab_flat.update(tokens)
    print(f"Vocabulary size (full): {len(vocabulary)}")
    print(f"Unique tokens: {len(vocab_flat)}")
    return vocabulary, vocab_flat


def filter_query_with_vocab(query, vocab_flat):

    query_tokens = re.findall(r"\w+", query.lower())
    filtered_tokens = [tok for tok in query_tokens if tok in vocab_flat]
    cleaned_query = " ".join(filtered_tokens)
    print("Filtered tokens:", filtered_tokens)
    print("Cleaned query:", cleaned_query)
    return cleaned_query, filtered_tokens


def extract_terms_from_query(query, vocabulary):
    # tokenize the query
    query_tokens = set(re.findall(r"\w+", query))
    if not query_tokens:
        return []

    matched_terms = []
    for term_tokens in vocabulary:
        term_set = set(term_tokens)
        # strict matching: all query tokens must be in term
        if query_tokens.issubset(term_set):
            matched_terms.append(term_tokens)

    return matched_terms



def extract_terms_fallback(query, vocabulary):
    # tokenize the query
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return []

    matched_terms = []
    for term_tokens in vocabulary:
        term_set = set(map(str.lower, term_tokens))
        # partial matching: at least one token overlaps
        if query_tokens & term_set:  
            matched_terms.append(term_tokens)

    return matched_terms



