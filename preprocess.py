import os
import cv2
import fitz  
import shutil
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw
from pathlib import Path
from img2table.document.image import Image as Img2TableImage
from img2table.ocr import DocTR
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from openpyxl import load_workbook
import pdfplumber
import re

# === Paths ===
pdf_folder = r"C:\Users\a5348634\OneDrive - Saint-Gobain\Documents\backup\Life Science\Design Recommender\pdfs"
output_dir = "output"
table_dir = os.path.join(output_dir, "tables")
cropped_dir = os.path.join(output_dir, "cropped_pages")
diagram_dir = os.path.join(output_dir, "diagrams")
masked_dir = os.path.join(output_dir, "diagrams(masked_pages)")
EXPECTED_HEADERS = [ 'item','qty.', 'description', 'material', 'stk. no.']

def normalize_header(header):
    return re.sub(r'\s+|\.', '', str(header).strip().lower())

def find_bom_table_from_excel(excel_path):
    xls = pd.ExcelFile(excel_path)
    
    for sheet_name in xls.sheet_names:
        #print(sheet_name)
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        for i, row in df.iterrows():
            row_values = row.fillna('').astype(str).tolist()
            norm_row = [normalize_header(cell) for cell in row_values]
            
            if all(any(expected in cell for cell in norm_row) for expected in [normalize_header(h) for h in EXPECTED_HEADERS]):
                #print(f"✅ BoM header found in sheet: {sheet_name}, row: {i}")
                
                # Extract table from this header row onward
                bom_df = pd.read_excel(xls, sheet_name=sheet_name, header=i)
                bom_df = bom_df[[col for col in bom_df.columns if str(col).strip().lower() in EXPECTED_HEADERS]]
                
                # Normalize columns
                bom_df.columns = [col.strip().lower() for col in bom_df.columns]
                #print(bom_df.columns)
                if bom_df.isna().all().all():
                    continue
                else:
                    return bom_df
            else:
                print("BoM headers not present in Excel path",excel_path)
    print("❌ BoM table not found in Excel path ",excel_path)
    return None

# === Cleanup & Create Directories ===
for d in [diagram_dir, table_dir, cropped_dir, masked_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)



# === Main Pipeline ===
for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]
    print(f"\n Processing PDF: {pdf_file}")
#From PDF Extract BOM    
    pdf = PDF(src=pdf_path)
    # Instantiation of the OCR, Tesseract, which requires prior installation
    ocr = TesseractOCR(lang="eng")
    #ocr =  DocTR(detect_language=False)    

    # Table identification and extraction
    pdf_tables = pdf.extract_tables(ocr=ocr)
    # We can also create an excel file with the tables
    pdf.to_xlsx('tablesnew.xlsx', ocr=ocr)
    bom_df = find_bom_table_from_excel("tablesnew.xlsx")
    # Normalize column names (optional but safer)
    if bom_df is not None:
        if 'item' in bom_df.columns:
            bom_df.columns = [col.strip().lower() for col in bom_df.columns]

            # Filter rows where 'item no.' is 
            df_filtered = bom_df[pd.to_numeric(bom_df['item'], errors='coerce').notnull()]

            # Optionally, convert 'item no.' to integers
            df_filtered['item'] = df_filtered['item'].astype(int)
            #df_filtered=bom_df
            #df_filtered.to_excel("BOM_Table.xlsx")
            #file_header = [pdf_name]
            #df_with_header = pd.concat([pd.DataFrame([file_header]), df_filtered], ignore_index=True)
            csv_path = os.path.join(table_dir, f"{pdf_name}_table.csv")
            df_filtered.to_csv(csv_path, index=False)
        else:
            with pdfplumber.open(pdf_path) as pdf:
                page1=pdf.pages[0]
                tables = page1.extract_tables()
            # Separate header and rows
            header = tables[0][0]
            rows = tables[0][1:]
            # Create DataFrame
            df = pd.DataFrame(rows, columns=header)
            # Apply transformation: keep only content after the first new line character
            df = df.applymap(lambda x: x.split("\n", 1)[-1] if isinstance(x, str) and "\n" in x else x)
            csv_path = os.path.join(table_dir, f"{pdf_name}_table.csv")
            df.to_csv(csv_path,index=False)    

    # === OCR Initialization ===
    ocr =  DocTR(detect_language=False)    
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            if page_num == 1:
                continue  # Skip page 2

            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cropped_img = img[:int(img.shape[0] * 0.78), :]
            cropped_path = os.path.join(cropped_dir, f"{pdf_name}.png")
            cv2.imwrite(cropped_path, cropped_img)
            print(f" Cropped and saved: {cropped_path}")

            try:
                doc_img = Img2TableImage(src=cropped_path)

                tables = doc_img.extract_tables(
                    ocr=ocr,
                    implicit_rows=True,
                    borderless_tables=True
                )

                if not tables:
                    print(" No tables detected.")
                    continue

                seen_tables = set()
                mask_img = PILImage.open(cropped_path).convert("RGB")
                draw = ImageDraw.Draw(mask_img)

                #saved_csv_paths = []  

                for idx, table in enumerate(tables[:2]):
                    if idx >= 2:
                        break
                    bbox_hash = tuple(
                        (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)
                        for row in table.content.values()
                        for cell in row if cell is not None
                    )
                    if bbox_hash in seen_tables:
                        print(" Duplicate table skipped.")
                        continue
                    seen_tables.add(bbox_hash)

                    # Save table CSV
                    table_data = []
                    max_cols = max(len(row) for row in table.content.values())

                    for row_idx in sorted(table.content.keys()):
                        row = table.content[row_idx]
                        row_data = [cell.value if cell else "" for cell in row]
                        row_data += [""] * (max_cols - len(row_data)) 
                        table_data.append(row_data)

                    df = pd.DataFrame(table_data)
                    file_header = [pdf_name] + [""] * (max_cols - 1)
                    df_with_header = pd.concat([pd.DataFrame([file_header]), df], ignore_index=True)
                    csv_path = os.path.join(table_dir, f"{pdf_name}_table_{idx + 1}.csv")
                    #df_with_header.to_csv(csv_path, index=False, header=False)
                    #saved_csv_paths.append(csv_path)
                    #print(f" Saved table CSV: {csv_path}")

                    # === Mask table areas ===
                    for row in table.content.values():
                        for cell in row:
                            if cell is None:
                                continue
                            x1 = int(cell.bbox.x1)
                            y1 = int(cell.bbox.y1)
                            x2 = int(cell.bbox.x2)
                            y2 = int(cell.bbox.y2)
                            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

                # Save masked + diagram image
                masked_path = os.path.join(masked_dir, f"{pdf_name}_masked.png")
                cropped_diagram = os.path.join(diagram_dir, f"{pdf_name}_diagram.png")
                mask_img.save(masked_path)
                mask_img.save(cropped_diagram)
                print(f" Masked table image saved: {masked_path}")
                print(f" Diagram cropped and saved: {cropped_diagram}")
            except Exception as e:
                print(f" Error processing page {page_num}: {e}")

print("\n All PDFs processed.")

'''
                # === Delete CSV if ALL second row cells are "revisions" ===
                for csv_path in saved_csv_paths:
                    try:
                        df_check = pd.read_csv(csv_path, header=None)

                        if len(df_check) > 1:
                            header_row = df_check.iloc[1].astype(str).str.strip().str.lower().tolist()
                            print(f" Checking headers in {csv_path}: {header_row}")

                            # Check if all values in the row are exactly "revisions"
                            if any(cell == "revisions"  or cell == "rev." for cell in header_row):
                                os.remove(csv_path)
                                print(f" Deleted CSV with only 'revisions': {csv_path}")
                            else:
                                print(f" Kept CSV: {csv_path}")
                                import os
                                # Now you want to update its name
                                new_csv_path = os.path.join(table_dir, f"{pdf_name}.csv")

                                # Rename it
                                os.rename(csv_path, new_csv_path)

                        else:
                            print(f" Not enough rows to check headers in {csv_path}")

                    except Exception as e:
                        print(f" Error checking CSV headers in {csv_path}: {e}")
'''
