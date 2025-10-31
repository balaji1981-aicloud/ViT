import os
import pandas as pd
import ollama
from ollama import Client
from tabulate import tabulate


client = Client(host='http://10.87.60.30:11434')


path = r"C:\Projects\HR Data Analysis\Survey Summary\Output\Function"
OUTPUT_FOLDER = r"C:\Projects\HR Data Analysis\Survey Summary\summaries\New"
MODEL = "gemma3:27b"

def summarize_excel(df1,label):
    # Load Excel
    df = df1.copy()
    
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(df.head())
    print(df.columns)
    # Check required columns
    required = {"questions", "response", "difference_from_2024"}#question", "response", "difference_from_last_year"}
    # if not required.issubset(set(df.columns)):
    #     raise ValueError(f"{file_path} missing required columns: {required}")

    # Compute quick stats
    avg_response = df["response"].mean()
    avg_diff = df["difference_from_2024"].mean()
    improved = df[df["difference_from_2024"] > 0]["questions"].tolist()
    declined = df[df["difference_from_2024"] < 0]["questions"].tolist()

    # Convert table to markdown
    table_text = tabulate(df.head(15), headers="keys", tablefmt="github")
    # label = "Technical Production"#os.path.basename(file_path).replace(".xlsx", "")

    # Prompt for the model
    prompt = f"""
You are a Gen AI Expert.
The following dataset is from the **{label}** segment of a employee satisfaction survey.
Each question includes its positive response percentage value and difference from last year.

Write a professional structured summary with sections:
1. Overall trend and performance
2. Improvements (areas that increased)
3. Declines (areas that decreased)
4. Positives / strengths
5. Negatives / concerns
6. Areas to focus next

Quick stats:
- Average response: {avg_response:.2f}
- Average change from last year: {avg_diff:+.2f}

Questions that improved: {improved if improved else 'None'}
Questions that declined: {declined if declined else 'None'}

example response:
  1. Overall Trend & Performance:  
Average response is high (89.25%), but trending downwards (-1.34% change). While generally positive, declines require attention to prevent further erosion of satisfaction.
 
  2. Improvements (Areas Increased):  
Work/life balance, future tenure, management trust and pace of change all saw increases. These suggest positive shifts in employee perception of support and company direction.
 
  3. Declines (Areas Decreased):  
Significant declines in recommendation (eNPS), respect, training, and feelings of value are concerning. These areas indicate potential systemic issues impacting morale and engagement.
 
  4. Positives / Strengths:  
High scores in health & safety (96%) and confidence in achieving objectives (94%) demonstrate commitment to core values and operational effectiveness. These are solid foundations to build upon.
 
  5. Negatives / Concerns:  
Multiple declines in areas related to employee value, recognition and development suggest a potential disconnect between expectations and reality. Declining eNPS is a critical warning sign.
 
6. Areas to Focus Next:
Prioritize initiatives addressing respect, training, and employee recognition. Investigate drivers behind declining eNPS. Strengthen communication about career opportunities and personal development pathways.


Keep it short: maximum 2-3 bullets per section, 20–30 words per section
survey data:
{table_text}
"""

    # Generate summary via Ollama Python client
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    summary = response.message.content

    # Save to file
    output_file = os.path.join(OUTPUT_FOLDER, f"{label}_summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ Summary created for {label}: {output_file}")
    return label, summary




for path ,dirr,file in os.walk(path):
    files = file

for i, file in enumerate(files):
    print(file)
    df1 = pd.read_excel(path+"//"+file)
    df2 = df1.copy()
    df2 = df2[['Questions', 'Response','Difference from 2024']]
    df2['Response'] = df2['Response'].replace('-',)
    df2['Difference from 2024'] = df2['Difference from 2024'].replace('-',)
    df2['Response'] = df2['Response'].astype(float)
    df2['Difference from 2024'] = df2['Difference from 2024'].astype(float)
    df2 = df2.groupby("Questions").mean().reset_index()
    summarize_excel(df2,file)


