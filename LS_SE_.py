import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os
import traceback
import warnings
from PIL import Image
from typing import List
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN,HDBSCAN
import scipy.cluster.hierarchy as sch
import re
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
# from marker.processors.table import TableProcessor
import torch
import seaborn as sns
import gc

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import logging
from datetime import datetime

os.makedirs("static", exist_ok=True)

# converter = TableConverter(artifact_dict=create_model_dict())

warnings.filterwarnings("ignore")

EXCEL_FILE = "user_log.xlsx"

def get_remote_ip():
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None

        return session_info.request.remote_ip
    except Exception:
        return None

def log_user_to_excel(ip, timestamp, app_name):
    new_entry = pd.DataFrame([[ip, timestamp, app_name]], columns=["IP", "Timestamp", "Application"])
    
    if os.path.exists(EXCEL_FILE):
        existing = pd.read_excel(EXCEL_FILE)
        updated = pd.concat([existing, new_entry], ignore_index=True)
    else:
        updated = new_entry

    updated.to_excel(EXCEL_FILE, index=False)


def logger_main(app_name="Assembly Configurator"):

    user_ip = get_remote_ip() or "Unknown"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_user_to_excel(user_ip, current_time, app_name)

# Configure memory settings
def configure_memory():
    """Configure memory settings for better resource management"""
    # Free unused memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Set lower precision to reduce memory usage
    torch.set_default_dtype(torch.float32)
    
    # Set smaller chunk size for model loading
    os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Prevent downloading models repeatedly
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

@st.cache_resource(show_spinner="Loading models...")
def initialize_models():
    """Initialize models with memory optimization"""
    try:
        configure_memory()
        
        # Initialize converter with lower precision
        converter = TableConverter(
            artifact_dict=create_model_dict(
                device='cpu',  # Force CPU usage
                dtype=torch.float32  # Use float32 instead of float64
            )
        )
        return converter
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None


st.set_page_config(layout="wide")
# Initialize models only if not already in session state
# if 'converter' not in st.session_state:
#     with st.spinner("Initializing models (this may take a moment)..."):
#         converter = initialize_models()
#         if converter is not None:
#             st.session_state.converter = converter
#         else:
#             st.error("Failed to initialize models. Please try increasing your system's virtual memory.")
#             st.stop()
# converter = st.session_state.converter

def process_bom_line(parts):
    # print(parts,"#####################")
    """
    Process a BOM line with special handling for missing values.
    
    Args:
        parts (list): List containing [Item_No, Part_Number, Description, Material, Quantity]
        
    Returns:
        dict: Processed BOM line with filled missing values
    """
    try:
        # First, handle cases where Item_No and Description are combined with <br>
        if len(parts) >= 2 and '<br>' in parts[1]:
            item_no, description = parts[1].split('<br>', 1)
            parts[0] = item_no
            parts[1] = ''  # Clear part number since it's empty
            parts[2] = description

        # Ensure we have at least 4 parts and first element is a number
        if len(parts) < 4 or not parts[0].strip().isdigit():
            return None

        # Clean parts by stripping whitespace
        parts = [p.strip() for p in parts]
        
        # Handle Part Number
        part_number = parts[1]
        description = parts[2]
        if not part_number:  # If Part Number is empty
            part_number = description.split()[0] if description else ''
            # Remove the used part from description
            if part_number:
                description = ' '.join(description.split()[1:])
        
        # Handle Material and Quantity
        material = parts[3]
        quantity = parts[4] if len(parts) > 4 else ''
        
        if not quantity:  # If Quantity is empty
            material_parts = material.split()
            
            if len(material_parts) >= 2:
                # Check if material ends with 'mm'
                if material_parts[-1].lower() == 'mm':
                    # Take last two parts if available
                    quantity = ' '.join(material_parts[-2:])
                    # Remove the used parts from material
                    material = ' '.join(material_parts[:-2])
                else:
                    # Take just the last part
                    quantity = material_parts[-1]
                    # Remove the used part from material
                    material = ' '.join(material_parts[:-1])
            else:
                quantity = material
                material = ''
        
        item = {
            'Item_No': parts[0],
            'Part_Number': part_number,
            'Description': description,
            'Material': material,
            'Quantity': quantity
        }
        
        return item
    except:
        pass

def clean_material_quantity(df, material_col="MATERIAL", qty_col="QTY."):
    """
    Cleans the QTY column by extracting any leading non-numeric prefixes 
    and appending them to the MATERIAL column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    material_col (str): The column name containing material names.
    qty_col (str): The column name containing quantities.

    Returns:
    pd.DataFrame: A modified DataFrame with cleaned MATERIAL and QTY. columns.
    """
    # df = df.copy()

    # Ensure QTY column is string and strip spaces
    df[qty_col] = df[qty_col].astype(str).str.strip()

    # Extract leading non-numeric characters (prefix) and numeric quantity
    df[["Prefix", qty_col]] = df[qty_col].str.extract(r"^(\D*)\s*(\d+.*)?$")
    
    # Fill NaN values with empty strings
    df["Prefix"] = df["Prefix"].fillna("").str.strip()
    df[qty_col] = df[qty_col].fillna("").str.strip()

    # Append the prefix to MATERIAL if it exists
    df[material_col] = df.apply(lambda x: f"{x[material_col]} {x['Prefix']}" if x["Prefix"] else x[material_col], axis=1)

    # Drop the temporary Prefix column
    df = df.drop(columns=["Prefix"])

    return df


def remove_leading_empty_strings(lst):
    while lst and lst[0] == "":
        lst.pop(0)
    return lst


def parse_bom_data(text):
    # Split into lines
    lines = text.split('\n')
    
    # Initialize results list
    bom_items = []
    
    # Find the header line to identify column positions
    header_line = None
    for line in lines:
        if 'ITEM NO.' in line and 'PART NUMBER' in line:
            header_line = line
            break
    
    if not header_line:
        return pd.DataFrame()  # Return empty DataFrame if no header found
            
    # Process lines that contain actual BOM data
    for line in lines:
        # Skip header and separator lines
        if '---' in line or 'ITEM NO.' in line:
            continue
            
        # Split by pipe and clean up each field
        parts = [part.strip() for part in line.split('|')]

        parts = remove_leading_empty_strings(parts)

        item = process_bom_line(parts)

            # print(item)
            
            # # Handle <br> tags in any field
            # for key in item:
            #     item[key] = item[key].replace('<br>', ' ').strip()
            
            # # Only add non-empty items
        try:
            if item['Item_No'] or item['Part_Number']:
                bom_items.append(item)
        except TypeError:
            continue
            

    # Convert to DataFrame
    df = pd.DataFrame(bom_items)
    
    # Convert Item_No to integer
    df['Item_No'] = pd.to_numeric(df['Item_No'], errors='coerce')
    
    # Sort by Item_No
    df = df.sort_values('Item_No')
    
    # Reset index
    df = df.reset_index(drop=True)

    df.rename({'Item_No': 'ITEM NO.',
            'Part_Number': "PART NUMBER",
            'Description': 'DESCRIPTION',
            'Material': 'MATERIAL',
            'Quantity': "QTY."},axis=1,inplace=True)
    
    return df

def process_bom_text(text):
    # Parse and convert to DataFrame
    df = parse_bom_data(text)
    
    return df

def extract_tables_from_pdf(pdf_path):# str) -> List[List[List[str]]]:
    """
    Extract tables from a PDF file using pdfminer.six.
    
    Parameters:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    List[List[List[str]]]: A list of tables, where each table is a list of rows, and each row is a list of cell values.
    """
    tables = []
    
    # pdf_bytes = pdf_path.read()pdf_bytes
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables_on_page = page.extract_tables()
            tables.extend(tables_on_page)
    
    return tables

def preprocess_table(table: List[List[str]]) -> List[List[str]]:
    """
    Preprocess a table by removing empty rows and columns, and adding a 'Quantity' column if it's missing.
    
    Parameters:
    table (List[List[str]]): A table extracted from a PDF.
    
    Returns:
    List[List[str]]: The preprocessed table.
    """
    preprocessed_table = []
    
    for row in table:
        if any(cell.strip() for cell in row):
            # if len(row) == 5:
            #     row.insert(4, 'Quantity')
            preprocessed_table.append([cell.strip() for cell in row])
    
    return preprocessed_table

def extract_bom_from_pdf(pdf_path, excel_path,folder_path):
    
    # print(pdf_path)/
    pdf_bytes = pdf_path.read()
    try:
        dataframes = [] 
        # Open the PDF file
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[0]
            table = page.extract_table({
            "intersection_x_tolerance": 3,
            "intersection_y_tolerance": 3})
            # Convert the table to a DataFrame
            df = pd.DataFrame(table)

            # Drop columns with any NaN values
            df.replace({None:np.nan,"":np.nan},inplace=True)
            df.dropna(inplace=True,axis=0,thresh=4)
            df.dropna(inplace=True,axis=1)

            # Append the cleaned DataFrame to the list if it's not empty
            if not df.empty:
                dataframes.append(df)


        # If you need to concatenate all the DataFrames into a single DataFrame
        if dataframes:
            final_df = pd.concat(dataframes, ignore_index=True)
            final_df.columns = final_df.iloc[0]
            final_df.drop([0],axis=0,inplace=True)
            final_df.reset_index(drop=True,inplace=True)

            final_df.dropna(how='any',axis=0,inplace=True)

            
            result_df = final_df
            try:
                result_df.columns = ['ITEM','QTY.','DESCRIPTION','MATERIAL','STK. NO.']#['ITEM NO.','PART NUMBER','DESCRIPTION','MATERIAL','QTY.']
                result_df['Filename'] = pdf_path.name

                result_df = clean_material_quantity(result_df)



                result_df['QTY.'] = result_df['QTY.'].apply(lambda x: 1 if any(c.isalpha() for c in str(x)) else x)
                result_df.reset_index(drop=True,inplace=True)

                if not result_df.empty:
                    result_df.to_excel(excel_path, index=False)
                
            except:
                tables = extract_tables_from_pdf(pdf_path)
                
                for table in tables:
                    try:
                        preprocessed_table = preprocess_table(table)
                        result_df = pd.DataFrame(preprocessed_table)
                        result_df.columns = result_df.iloc[0]
                        result_df.drop([0],axis=0,inplace=True)
                        result_df.reset_index(drop=True,inplace=True)
                        

                        result_df.dropna(how='any',axis=0,inplace=True)
                        # print(result_df)
                        result_df['Filename'] = pdf_path.name

                        result_df = clean_material_quantity(result_df)


                        result_df['QTY.'] = result_df['QTY.'].apply(lambda x: 1 if any(c.isalpha() for c in str(x)) else x)
                

                        result_df.reset_index(drop=True,inplace=True)
                        if not result_df.empty:
                            result_df.to_excel(excel_path, index=False) 
                    except:
                        break

    except:
        print(pdf_path.name,"********************************************************")
           

def crop_white_spaces(image):
    """
    Remove rows and columns that are entirely white
    
    Args:
    image (PIL.Image): Input image to process
    
    Returns:
    PIL.Image: Image with all-white rows and columns removed
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Check if entire rows are white
    white_rows = np.all(np.all(img_array == [255, 255, 255], axis=2), axis=1)
    
    # Check if entire columns are white
    white_cols = np.all(np.all(img_array == [255, 255, 255], axis=2), axis=0)
    
    # Remove white rows
    img_array_filtered = img_array[~white_rows]
    
    # Remove white columns
    img_array_filtered = img_array_filtered[:, ~white_cols]
    
    # Convert back to PIL Image
    return Image.fromarray(img_array_filtered)

def morph_pdf_tables_to_white_and_crop(input_pdf_path):#, output_image_path):
    """
    Morph tables in a PDF to white color, convert to image, and crop white spaces
    
    Args:
    input_pdf_path (str): Path to the input PDF file
    output_image_path (str): Path to save the output image
    """
    # Open the PDF document
    doc = fitz.open(input_pdf_path)
    
    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Find tables on the page
        tables = page.find_tables()
        
        # Morph each table to white
        for table in tables:
            # Get the table's bbox (bounding box)
            bbox = table.bbox
            
            # Create a white drawing using the bounding box
            page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1), width=1)
    
    # Render page to a pixmap (image)
    page = doc[0]  # Assumes we want the first page
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # High-resolution matrix
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Crop white spaces
    cropped_img = crop_white_spaces(img)
    
    # Save the image
    return cropped_img
 
def get_table_positions(pdf_path, page_number):
    table_positions = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
 
   
        # Extract tables from the PDF
        tables = page.extract_tables({
        "intersection_x_tolerance": 3,
        "intersection_y_tolerance": 3})
        tables_1 = page.find_tables({
        "intersection_x_tolerance": 3,
        "intersection_y_tolerance": 3})
        for i in range(0,len(tables)):
            if len(tables[i]) > 0:  # Ensure the table is not empty
                df = pd.DataFrame(tables[i][1:], columns=tables[i][0])
                if df.columns[0]!=None:
                    # list(df.columns).['ITEM NO.', 'PART NUMBER', 'DESCRIPTION', 'MATERIAL', 'QTY.'])
                    if (any(element in list(df.columns) for element in ['ITEM','QTY.','DESCRIPTION','MATERIAL','STK. NO.'])) or ('CUSTOMER NAME' in df.columns[0]):# or ('ASSEMBLED' in df.columns[0]):
                     
                        bbox = tables_1[i].bbox  # bbox format: (x0, top, x1, bottom)
                        table_positions.append(bbox)
                        print(table_positions)

    return table_positions

########################## END OF TABLE EXTRACTION ###########################################
 
def render_page_as_image(pdf_path, page_number, zoom=1):
    pdf_document = fitz.open(pdf_path)

    page = pdf_document.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
   
    if pix.n == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
    return image                  
 
def crop_drawing(image, table_positions):
    # print(image.shape)
    if len(table_positions) >= 2:
        top_table = table_positions[0]
        bottom_table = table_positions[1]
        ax0,ay_top , ax1, ay_bottom = top_table
        bx0,by_top , bx1, by_bottom = bottom_table #(666.8973451327436, 364.82631578947365, 678.826515151514, 417.90000000000003)#bottom_table
        # print( ax0,ay_top , ax1, ay_bottom,bx0,by_top , bx1, by_bottom)     
        cropped_image = image[int(ay_bottom):int(by_top)-1, 0:int(bx1)]#int(ax0)-2
 
        return cropped_image
    else:
        # Image dimensions
        height, width, _ = image.shape

        # Calculate the cropping boundaries
        upper_boundary = int(height // 2.5)  # Start after the upper half
        lower_boundary = int(height - (height // 4.8))  # Exclude the lower quarter

        # Crop the middle quarter of the image
        cropped_image = image[upper_boundary:lower_boundary, :]

        return cropped_image


 
def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    # print(f"Image saved to {output_path}")  

def erase_red_color(image):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")
    
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to filter red regions
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Invert mask to keep non-red regions
    non_red_mask = cv2.bitwise_not(red_mask)

    # Apply mask to the image
    result_image = cv2.bitwise_and(image, image, mask=non_red_mask)

    return result_image

######################### END OF IMAGE EXTRACTOR ##############################
 
def search_for_columns(data, search_terms):
  """Searches for columns in a list of strings that contain all the specified search terms.

  Args:
    data: A list of strings representing the columns.
    search_terms: A list of strings to search for.

  Returns:
    A list of columns that contain all the search terms.
  """

  results = []
  for column in data:
    if all(term in column for term in search_terms):
      results.append(column)
  return results

# Function to increment "END CONNECTOR (Male, Female, Genderless Luers)" based on suffix columns
def increment_end_connector(row, suffix_columns):
    # Extract values from all suffix columns
    suffix_values = row[suffix_columns]
    
    if (suffix_values > 0).all():
        # Increment by 2 if all suffix columns have values > 0
        return row['END CONNECTOR (Male, Female, Genderless Luers)'] + 2
    elif (suffix_values == 0).all():
        # No increment if all suffix columns are 0
        return row['END CONNECTOR (Male, Female, Genderless Luers)']
    else:
        # Increment by 1 if any suffix column is 0 but not all
        return row['END CONNECTOR (Male, Female, Genderless Luers)'] + 1

def save_uploaded_file(uploaded_file, folder_path):
    """Saves the uploaded file to the specified folder."""
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the uploaded file
    file_path = os.path.join(folder_path, uploaded_file.name[:-3]+"pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

@st.cache_data
def remove_file(folder_path, uploaded_files):
    """Removes files in a folder, excluding those that match the uploaded files by name (ignoring extensions)."""
    # Extract the base names of the uploaded files (excluding extensions)
    uploaded_file_basenames = [os.path.splitext(file.name)[0] for file in uploaded_files]

    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Extract the base name of the current file in the folder
            folder_file_basename = os.path.splitext(filename)[0]
            try:
                # Skip deletion if the file name matches any uploaded file name
                if folder_file_basename in uploaded_file_basenames:
                    continue
                # Delete the file or directory
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")

# @st.cache_data  
def pdf_parser(folder_path,uploaded_file):
    try:
        folder_path = r"C:\Users\P6455771\OneDrive - Saint-Gobain\Backup\Banglore Deployment\PDF - Copy"
        # print(uploaded_file,uploaded_file.name)
        
        save_uploaded_file(uploaded_file, folder_path)

        pdf_file_path = uploaded_file
        image_output_path = os.path.join(folder_path,uploaded_file.name[:-3]+'jpg')
        excel_output_path = os.path.join(folder_path,uploaded_file.name[:-3]+'xlsx')

        # Extract BOM as Excel
        extract_bom_from_pdf(pdf_file_path, excel_output_path,folder_path)
       
        cropped_drawing = morph_pdf_tables_to_white_and_crop(os.path.join(r'C:\Users\P6455771\Projects\IOT\LS Configurator Tool\84 Assemblies\PDF - Copy',uploaded_file.name[:-3]+"pdf"))
        cropped_drawing.save(image_output_path)

    except Exception as e:
        print("#")
        print(traceback.format_exc())


def resize_image(image: Image.Image) -> Image.Image:
    """
    Resize image to fit within target size while preserving aspect ratio and preventing cropping
    
    Args:
        image: PIL Image object
        target_size: Target width and height
    Returns:
        Resized PIL Image object
    """
    target_size = (400, 300)
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    if original_aspect > target_aspect:
        # Image is wider than target
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        # Image is taller than target
        new_height = target_height
        new_width = int(target_height * original_aspect)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def display_image_with_checkbox(image_path: str, key: str, caption: str, location: str = "main") -> bool:
    """
    Display an image with a checkbox and handle errors
    
    Args:
        image_path: Path to the image file
        key: Unique key for the checkbox
        caption: Caption for the image
        location: Either "main" or "sidebar"
    Returns:
        Boolean indicating if checkbox is checked
    """
    try:
        image = Image.open(image_path)
        resized_image = resize_image(image)
        
        if location == "sidebar":
            with st.sidebar:
                is_checked = st.checkbox(f"Select {key}", key=f"sidebar_{key}")
                st.image(resized_image, caption=caption, use_column_width=True)
        else:
            col1, col2 = st.columns([0.2, 0.8])
            with col1:
                is_checked = st.checkbox(key, key=f"main_{key}")
            with col2:
                st.image(resized_image, caption=caption, use_column_width=True)
        
        return is_checked
    except Exception as e:
        error_msg = f"Error loading image for {key}: {str(e)}"
        if location == "sidebar":
            st.sidebar.error(error_msg)
        else:
            st.error(error_msg)
        return False

# Apply the styles once
st.markdown("""
    <style>
        .stImage {
            position: relative;
            width: 100%;
            min-height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: visible !important;
        }
        
        .stImage > img {
            max-width: 100% !important;
            max-height: 300px !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

def add_footer():
    """Add a footer to the Streamlit app"""
    footer = """
    <style>
        footer {
            visibility: hidden;
        }
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f2f6;
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #ccc;
        }
        .custom-footer p {
            margin: 0;
            font-size: 0.9rem;
            color: #666;
        }
    </style>
    <div class="custom-footer">
        <p> Developed by SGRI Data Analytics Team</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)



######################################  MAIN CODE ###############################################


# Streamlit App
st.title("Assembly Configurator")
# Center align the title using markdown with HTML
st.markdown(
    "<h1 style='text-align: center;'>The DREAM Tool</h1>",
    unsafe_allow_html=True
)

# Center align the abbreviation with a smaller font
st.markdown(
    "<h3 style='text-align: center; font-size: 25px;'>Design Recommendation Engine for Assemblies using Master Models</h3>",
    unsafe_allow_html=True
)

# Center align the abbreviation with a smaller font
st.markdown(
    "<h3 style='text-align: left; font-size: 30px;'>Assembly Configurator</h3>",
    unsafe_allow_html=True
)

# Sidebar: Upload PDFs
st.sidebar.title("Upload Assemblies")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files for assemblies", accept_multiple_files=True, type=["pdf"]
)

# Process and display extracted images
if True:
    extracted_images = {}
    folder_path = r"C:\Users\P6455771\OneDrive - Saint-Gobain\Backup\Banglore Deployment\PDF - Copy"
    remove_file(folder_path,uploaded_files)
    for uploaded_file in uploaded_files:
        pdf_parser(folder_path,uploaded_file)
if True:
    # try:
    # Define the directory path where the files are located
    logger_main()
    
    directory_path = r"C:\Users\P6455771\OneDrive - Saint-Gobain\Backup\Banglore Deployment\PDF - Copy"

    # Initialize lists to store the paths of Excel and PNG files
    excel_files = []
    png_files = []

    # Walk through the directory to find Excel and PNG files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.xlsx'):  # Check for Excel files
                excel_files.append(os.path.join(root, file))
            elif file.endswith('.jpg'):  # Check for PNG files
                png_files.append(os.path.join(root, file))

    # Now, map the Excel files and PNG files (example: based on their names or order)
    mapped_files = {}

    count = 0
    # Assume we want to map files by their names (without extension)
    for excel_file in excel_files:
        # Extract the base name without extension
        base_name = os.path.splitext(os.path.basename(excel_file))[0]  # This gets name without extension

        # Check for matching PNG file by name
        png_file = next(
            (png for png in png_files if os.path.splitext(os.path.basename(png))[0] == base_name), 
            None
        )
        
        if png_file:
            mapped_files[base_name] = {'excel': excel_file, 'png': png_file}

        if count==0:
            data = pd.read_excel(excel_file)
            count=1
        else:
            data = data._append(pd.read_excel(excel_file))
       
   