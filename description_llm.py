import csv
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import re
import urllib3

# Input & output directories
INPUT_DIR = "./output/tables"
OUTPUT_DIR = "./output/table_updated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Ollama
llm = Ollama(model="qwen3:latest")

# Prompt template to process the whole table
DETAIL_PROMPT = PromptTemplate(
    input_variables=["headers", "rows"],
    template=(
        "You are an expert technical assistant.\n"
        "You will receive a table with headers and rows.\n"
        "For each row, create a short and precise description, without any punctuation used (e.g., \",\" etc), needed only one single continuous line description \n"
        "based on the DESCRIPTION column. Focus only on DESCRIPTION column. Focus on adding details that help understand \n"
        "dimensions and measurements from the units and symbols in the description. \n"
        "Return the output as a CSV table with an extra column 'DESCRIPTION_DETAILS', \n"
        "keeping all original headers and data unchanged.\n\n"
        "HEADERS:\n{headers}\n\n"
        "ROWS:\n{rows}\n\n"
        "CSV OUTPUT: \n"
    )
)

def enrich_table_with_llm(headers, rows):
    try:
        rows_str = "\n".join([",".join(row) for row in rows])
        prompt = DETAIL_PROMPT.format(headers=",".join(headers), rows=rows_str)
        enriched_csv_text = llm.invoke(prompt)
        enriched_csv_text = re.sub(r"<think>.*?</think>", "", enriched_csv_text, flags=re.DOTALL).strip()
        enriched_rows = []
        for line in enriched_csv_text.strip().splitlines():
            enriched_rows.append([cell.strip() for cell in line.split(",")])
        return enriched_rows
    except Exception as e:
        print(f"[ERROR] Failed to enrich table: {e}")
        return [headers] + rows

def process_csv_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if len(rows) < 3:
        print(f"[WARN] Skipping {file_path} (not enough rows)")
        return

    title = rows[0]
    headers = rows[1]      
    data_rows = rows[2:]    

    print(f"[INFO] Processing '{file_path}' → {len(data_rows)} rows")

    enriched_rows = enrich_table_with_llm(headers, data_rows)

    # Save updated CSV
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(title)
        writer.writerows(enriched_rows)
    print(f"[INFO] Saved enriched CSV: {output_path}")

def main():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    for csv_file in csv_files:
        process_csv_file(os.path.join(INPUT_DIR, csv_file))

if __name__ == "__main__":
    main()
