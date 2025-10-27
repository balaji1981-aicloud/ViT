
import os
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==============================
# CONFIGURATION
# ==============================
DATA_FOLDER = r"C:\Users\a5348634\OneDrive - Saint-Gobain\Documents\backup\Life Science\Design Recommender\output\tables"          # Folder containing your CSVs
OUTPUT_FOLDER = r"C:\Users\a5348634\OneDrive - Saint-Gobain\Documents\backup\Life Science\Design Recommender\output\tables_dimensions"     # Output folder for enriched CSVs
MODEL_NAME = "qwen3:4b"             # Ollama model name (ensure it’s pulled)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 📂 1️⃣ Folder containing all your CSVs (each CSV = one assembly)

# 🧠 2️⃣ Initialize Local LLM (Qwen or Llama)
llm = ChatOllama(
    model="phi4:latest",  # you can also use "qwen2.5:14b", "llama3:8b", etc.
    temperature=0.0,
)

# 🧾 3️⃣ Prompt Template for Dimension Extraction
prompt = ChatPromptTemplate.from_template("""
    You are an expert in life science single-use Assemblies and components.
    Explain this life science component of a single use assembly in a crisp single sentence capturing its dimensions and capacity if available from the below given component description.
    Component description: {description}
    Dimensions may be ID,OD,thickness etc.Capacity may be volume in litres etc.HB means "Hose Barb" for life science components.
    If dimensions and capacity are not mentioned in the component description,do not mention dimension and capacity.
    Restrict answer strictly to one sentence.
    Strictly do not add any extra Notes,spaces or explanations before and after.
Few shot examples:
              The TUBING C-FLEX® 374 is a single-use tubing with an internal diameter of 0.125 inches (3.2mm) and an external diameter of 0.250 inches (6.4mm). 
                The BAG CHAMBER - BAG BLANK 5L is a single-use life science component with a capacity of 5 liters.
                The 8" HANDLE SUPPORT BAR has an outer diameter of 5/16 inches.  
""")

# ⚙️ 4️⃣ Iterate through all CSV files and enrich them
for filename in os.listdir(DATA_FOLDER):
    if not filename.endswith(".csv"):
        continue

    csv_path = os.path.join(DATA_FOLDER, filename)
    df = pd.read_csv(csv_path)

    if "description" not in df.columns:
        print(f"⚠️ Skipping {filename}: no 'description' column found.")
        continue

    extracted_dims = []

    print(f"\n📄 Processing {filename} ...")

    for desc in tqdm(df["description"], desc=f"Extracting from {filename}"):
        try:
            # Prepare LLM input
            messages = prompt.format_messages(description=desc)
            response = llm.invoke(messages)
            text = response.content.strip()
            dim_value=text
        except Exception as e:
            dim_value = f"Error: {e}"

        extracted_dims.append(dim_value)

    # 🧩 Add new column and save enriched CSV
    df["dimension"] = extracted_dims
    enriched_path = os.path.join(OUTPUT_FOLDER, f"{filename}")
    df.to_csv(enriched_path, index=False, encoding="utf-8")

    print(f"✅ Saved enriched file: {enriched_path}")
