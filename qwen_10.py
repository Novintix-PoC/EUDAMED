import streamlit as st
import msal
import requests
import zipfile
import io
import pandas as pd
import tempfile
import os 
import base64
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import time
from pathlib import Path
import re
import numpy as np
from streamlit import cache_data
import cv2
import subprocess
from docx2pdf import convert as docx_to_pdf
# from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, set_seed, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

# Custom Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp {
            background: linear-gradient(to top, #d6eaff 50%, #aad4ff 80%);
            font-family: 'Roboto', sans-serif;
        }
        header[data-testid="stHeader"] {
            background: #aad4ff;
            padding: 0px;
            border-radius: 0px;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        .header-container img {
            height: 55px;
        }
        .header-container .title {
            font-size: 40px;
            font-weight: bold;
            color: #f5a82c;
        }
    </style>
""", unsafe_allow_html=True)

header_html = """
<div class="header-container">
    <img src="https://novintix.com/wp-content/uploads/2023/12/logov2.png">
    <div class="title">EUDAMED-Data Extraction</div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify which GPUs to use
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

# Check available GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Clear GPU cache
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model name
model_name = "Qwen/Qwen2-VL-7B-Instruct"

@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    print("Loading model - this should appear only once")
    # Load model with explicit device handling
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-72B-Instruct",
    #     torch_dtype=torch.float16,
    #     device_map="cuda",  # Specify exact device instead of "auto"
    #     attn_implementation="flash_attention_2",
    #     low_cpu_mem_usage=True
    # )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",  # This will automatically distribute across available GPUs
        max_memory={0: "23GB", 1: "23GB"},  # Specify memory limits per GPU
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    )

    # Then move to GPU if it fits
    # model = model.to(device)
    
    # min_pixels = 256 * 28  # e.g. 7168
    # max_pixels = 1280 * 28  # e.g. 35840

    # # Convert pixel values into edge lengths (roughly)
    # processor = AutoProcessor.from_pretrained(
    #     "Qwen/Qwen2-VL-72B-Instruct-AWQ",
    #     size={"shortest_edge": 256, "longest_edge": 1280}
    # )


    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # Load processor
    # processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# Load environment variables
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
tenant_id = st.secrets["TENANT_ID"]
redirect_uri = st.secrets["URL"]

# Azure AD details
authority_url = f'https://login.microsoftonline.com/{tenant_id}'
scopes = ['Files.ReadWrite.All', 'Sites.Read.All', 'User.Read']

# MSAL app configuration
app = msal.ConfidentialClientApplication(
    client_id,
    authority=authority_url,
    client_credential=client_secret
)

# Authentication functions
def get_auth_url():
    return app.get_authorization_request_url(scopes, redirect_uri=redirect_uri)

def get_token_from_code(auth_code):
    return app.acquire_token_by_authorization_code(auth_code, scopes=scopes, redirect_uri=redirect_uri)

def get_auth_headers(auth_code=None):
    if auth_code:
        token_response = get_token_from_code(auth_code)
        if 'access_token' in token_response:
            return {'Authorization': f'Bearer {token_response["access_token"]}'}
        else:
            # Log the error response for debugging
            st.error(f"Token response error: {token_response}")
    return None

def read_part_numbers_from_excel(uploaded_file):
    """
    Extracts part numbers from an uploaded Excel file.
    
    Args:
        uploaded_file: The uploaded Excel file object
        
    Returns:
        List of part numbers extracted from the file
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # Try common column names for part numbers
        potential_columns = ['Part Number', 'PartNumber', 'Part_Number', 'Part#', 'PN', 
                            'part number', 'partnumber', 'part_number', 'part#', 'pn']
        
        # Find the first matching column
        part_number_col = None
        for col in potential_columns:
            if col in df.columns:
                part_number_col = col
                break
        
        # If no standard column found, use the first column
        if part_number_col is None:
            part_number_col = df.columns[0]
        
        # Extract part numbers
        part_numbers = df[part_number_col].astype(str).tolist()
        
        # Clean up part numbers (remove NaN, empty strings, etc.)
        part_numbers = [str(pn).strip() for pn in part_numbers if str(pn).strip() and str(pn).lower() != 'nan']
        
        return part_numbers
    
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return []

# List all OneDrive files
def list_onedrive_files(folder_id=None, headers=None):
    files = []
    folder_url = "https://graph.microsoft.com/v1.0/me/drive/root/children" if not folder_id else \
                 f"https://graph.microsoft.com/v1.0/me/drive/items/{folder_id}/children"

    while folder_url:
        response = requests.get(folder_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            files.extend(data.get('value', []))
            folder_url = data.get('@odata.nextLink')
        else:
            st.error(f"Failed to fetch OneDrive files: {response.status_code}")
            break
    return files

# Recursive function to download OneDrive files
def download_onedrive_files_recursive(folder_id=None, folder_path="", headers=None, zip_file=None, part_number=None):
    """
    Recursively download files from OneDrive, filtered by part number if provided.
    
    Args:
        folder_id: ID of the current folder
        folder_path: Path to the current folder
        headers: Authentication headers
        zip_file: ZIP file to store downloaded files
        part_number: Part number to filter by (folder name)
    """
    files = list_onedrive_files(folder_id, headers)
    
    for file in files:
        file_name = file.get("name")
        file_id = file.get("id")
        
        # If part_number is specified, only process matching folders
        if part_number and file.get("folder"):
            if file_name.lower() != part_number.lower():
                # Skip this folder if it doesn't match the part number
                continue
        
        if file.get("folder"):
            new_folder_path = f"{folder_path}{file_name}/"
            download_onedrive_files_recursive(file_id, new_folder_path, headers, zip_file, None)  
            # Pass None as part_number since we're already inside the target folder
        else:
            file_content = download_onedrive_file(file_id, headers)
            if file_content:
                zip_file.writestr(f"{folder_path}{file_name}", file_content)

# Download a single file from OneDrive
def download_onedrive_file(file_id, headers):
    download_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"
    response = requests.get(download_url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download file {file_id}: {response.status_code}")
        return None

# List accessible SharePoint sites
def list_accessible_sites(headers):
    sites_url = "https://graph.microsoft.com/v1.0/sites?search=*"
    sites = []
    next_link = sites_url
    while next_link:
        response = requests.get(next_link, headers=headers)
        if response.status_code == 200:
            data = response.json()
            sites.extend([(site['id'], site['name']) for site in data.get('value', [])])
            next_link = data.get('@odata.nextLink')
        else:
            st.error(f"Failed to fetch sites: {response.status_code}")
            break
    return sites

def list_document_libraries(site_id, headers):
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    libraries = []
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        drives = response.json().get("value", [])
        for drive in drives:
            libraries.append({
                'id': drive.get('id'),
                'name': drive.get('name'),
                'driveType': drive.get('driveType')
            })
    return libraries

# Get document library ID from a SharePoint site
def get_drive_id(site_id, headers):
    drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    response = requests.get(drive_url, headers=headers)
    if response.status_code == 200:
        drives = response.json().get("value", [])
        if drives:
            return drives[0]["id"]
    return None

# List all files in a SharePoint document library
def list_all_files(drive_id, folder_id=None, headers=None):
    files = []
    folder_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children" if folder_id else \
                 f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
    while folder_url:
        response = requests.get(folder_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            files.extend(data.get('value', []))
            folder_url = data.get('@odata.nextLink')
        else:
            st.error(f"Failed to fetch files: {response.status_code}")
            break
    return files

def search_for_folders_by_name(drive_id, part_number, headers, progress_text=None):
    matching_folders = []
    folders_to_check = [{"id": "root", "path": "/"}]
    visited_folders = set()

    while folders_to_check:
        current = folders_to_check.pop(0)
        folder_id = current["id"]
        current_path = current["path"]

        if folder_id in visited_folders:
            continue
        visited_folders.add(folder_id)

        if progress_text:
            progress_text.text(f"Searching in {current_path}...")

        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children" if folder_id == "root" else \
              f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            continue

        items = response.json().get("value", [])
        for item in items:
            if "folder" in item:
                name = item.get("name", "")
                item_id = item.get("id")
                new_path = f"{current_path}{name}/"

                if name.lower() == part_number.lower():
                    matching_folders.append({
                        "id": item_id,
                        "path": new_path
                    })
                folders_to_check.append({
                    "id": item_id,
                    "path": new_path
                })
    return matching_folders

def download_folder_contents(drive_id, folder_id, folder_path, headers, zip_file):
    folders_to_process = [{"id": folder_id, "path": folder_path}]
    while folders_to_process:
        current = folders_to_process.pop(0)
        current_id = current["id"]
        current_path = current["path"]

        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{current_id}/children"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            continue

        items = response.json().get("value", [])
        for item in items:
            name = item.get("name")
            item_id = item.get("id")
            if "folder" in item:
                folders_to_process.append({
                    "id": item_id,
                    "path": f"{current_path}{name}/"
                })
            else:
                download_url = item.get("@microsoft.graph.downloadUrl")
                if not download_url:
                    meta_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
                    meta_resp = requests.get(meta_url, headers=headers)
                    if meta_resp.status_code == 200:
                        download_url = meta_resp.json().get("@microsoft.graph.downloadUrl")

                if download_url:
                    file_resp = requests.get(download_url)
                    if file_resp.status_code == 200:
                        zip_file.writestr(f"{current_path}{name}", file_resp.content)

# Download a single file from SharePoint
def download_file(drive_id, file_id, headers):
    download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
    response = requests.get(download_url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download file {file_id}: {response.status_code}")
        return None

# File Conversion and Processing Functions
def convert_docx_to_pdf_linux(input_file, output_file):
    """Convert docx to pdf using LibreOffice on Linux"""
    try:
        output_dir = os.path.dirname(output_file)
        subprocess.run([
            'libreoffice', '--headless', '--convert-to', 'pdf',
            '--outdir', output_dir, input_file
        ], check=True)
        
        temp_pdf = os.path.join(output_dir, f"{Path(input_file).stem}.pdf")
        if temp_pdf != output_file:
            os.rename(temp_pdf, output_file)
        return True
    except Exception as e:
        print(f"Error converting DOCX to PDF: {str(e)}")
        return False

# File Conversion and Processing Functions
def convert_to_image(file_path, output_folder):
    """
    Converts document files to images and stores them in the output folder.

    Args:
        file_path: Path to the document file.
        output_folder: Folder to save converted images.

    Returns:
        List of image file paths.
    """
    file_ext = Path(file_path).suffix.lower()
    output_images = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        if file_ext == '.pdf':
            images = convert_from_path(file_path, poppler_path=None)
            for i, image in enumerate(images):
                img_path = os.path.join(output_folder, f"{Path(file_path).stem}_page_{i}.jpg")
                image = image.convert('RGB')  # Ensure RGB format
                image.save(img_path, 'JPEG', quality=95)
                output_images.append(img_path)

        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = Image.open(file_path)

            # Fix: Convert "P" (palette) mode images with transparency to "RGBA"
            if image.mode == "P":
                image = image.convert("RGBA")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            processed_path = os.path.join(output_folder, f"{Path(file_path).stem}_processed.jpg")
            image.save(processed_path, 'JPEG', quality=95)
            output_images.append(processed_path)

        elif file_ext == '.docx':
            pdf_path = os.path.join(output_folder, f"{Path(file_path).stem}.pdf")
            convert_docx_to_pdf_linux(file_path, pdf_path)
            return convert_to_image(pdf_path, output_folder)


    except Exception as e:
        print(f"Error converting file {file_path}: {str(e)}")
        return []

    return output_images

def convert_all_files_with_retry(file_paths, output_folder, max_retries=3):
    """
    Retry image conversion for failed files until all are processed or retry limit is hit.
    """
    success_paths = []
    retry_files = file_paths.copy()
    
    for attempt in range(max_retries):
        if not retry_files:
            break
        current_retry = retry_files
        retry_files = []
        
        for file_path in current_retry:
            try:
                converted = convert_to_image(file_path, output_folder)
                if converted:
                    success_paths.extend(converted)
                else:
                    retry_files.append(file_path)
            except Exception as e:
                retry_files.append(file_path)
    
    if retry_files:
        st.warning(f"Some files could not be converted after {max_retries} retries: {retry_files}")
    
    return success_paths

# Apply the retry logic here:
with tempfile.TemporaryDirectory() as temp_dir:
    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    file_paths = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))

    image_paths = convert_all_files_with_retry(file_paths, images_dir)

def preprocess_image(image_path):
    """Enhanced image preprocessing using resizing, grayscaling, noise reduction, and unsharp masking."""
    try:
        # Open the image using PIL
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio and quality
        max_dimension = 2500
        min_dimension = 1000
        width, height = image.size

        if max(width, height) > max_dimension:
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        elif min(width, height) < min_dimension:
            ratio = min_dimension / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale using OpenCV
        img_array = np.array(image)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filtering for denoising
        denoised_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Apply unsharp masking using OpenCV
        gaussian_blur = cv2.GaussianBlur(denoised_image, (0, 0), 3)
        sharpened_image = cv2.addWeighted(denoised_image, 1.5, gaussian_blur, -0.5, 0)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(sharpened_image)
        
        # Save with high quality
        enhanced_path = f"{image_path}_enhanced.jpg"
        enhanced_image.save(enhanced_path, 'JPEG', quality=95, optimize=True)
        
        return enhanced_path
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return image_path

def encode_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# def enhanced_extract_with_vision(image_path, keyword):
def enhanced_extract_with_vision(image_path, keyword, model, processor):

    """
    Extracts information based on keywords from an enhanced image using Qwen2-VL-7B-Instruct model.

    Args:
        image_path: Path to the preprocessed image.
        keyword: The attribute keyword to search for.

    Returns:
        Extracted information or None if not found.
    """
    # Load the model and processor
    # model, processor = load_model()  # Now `model` is an actual model object, not a string

    try:
        enhanced_path = preprocess_image(image_path)  # Preprocess the image first
        is_yes_no = "(Y/N)" in keyword
        clean_keyword = keyword.replace("(Y/N)", "").strip()
        
        # Prepare the prompt
        if is_yes_no:
            prompt = f"""
                       Check if '{clean_keyword}' or related information for the '{clean_keyword}' exists in the documents.
                       Understand the '{clean_keyword}' in the input file and search for the '{clean_keyword}' in the files.
                       Interpret the content related to the '{clean_keyword}' in the files and return the correct 'Yes' or 'No' output based on that understanding of the presence of '{clean_keyword}'in the files.
                       Ensure the response is based only on the information present in the same file where the '{clean_keyword}'is identified.
                       Return only 'Yes' or 'No' based on the model's understanding of the content related to '{clean_keyword}' present in the files.
                       Return only 'Not Found' if neither the '{clean_keyword}' nor its related information is not mentioned in the files.
                       If the related information for '{clean_keyword}' is present in a table or structured format, the model should understand the content and still return the appropriate 'Yes' or 'No' output.
                       Do not explain your reasoning – just return 'Yes', 'No', or 'Not Found' as specified.
                       
                      """
        else:
            prompt = f"""
                       Check if the '{keyword}' is present in the documents.
                       Understand the '{keyword}' and extract the exact information related to the '{keyword}' in the files and return the correct information related to the '{keyword}'.
                       Understand the content in the file and extract the exact information alone from the contents in the file related to the '{keyword}'.
                       Return the whole specific text in the file that directly relates to '{keyword}' in the file.
                       Identify '{keyword}' in the exact file where it appears, and return the associated or descriptive information present on the same line or in the next line to the '{keyword}'.
                       If the input '{keyword}' is represented using a symbol or icon in the input documents, Understand the '{keyword}' that appears in symbol and extract the specific information related to the '{keyword}' such as meaning, label, or descriptive text associated with the '{keyword}'which is represented using symbol or icon identified in the input documents.
                       If '{keyword}' related information appears in a table or structured format also, understand the content and return the structured data accurately.
                       Ensure variations of '{keyword}' (abbreviations, singular/plural forms, synonyms, upper case/lower case) are also recognized.
                       If the '{keyword}' itself is not explicitly mentioned in the documents but the related information is present in the documents (like label), the model should understand the context and extract the relevant information related to the '{keyword}' from the documents.
                       Make sure the extracted information comes strictly from the exact file where the '{keyword}' or its symbolic/contextual reference is found. 
                       The model should understand the content of the files and extract the exact information related to the '{keyword}' from the exact file where the '{keyword}' is found. 
                       Do not extract the information which is not exactly related to the '{keyword}' from the unrelated files.
                       Do not return the '{keyword}' itself as the output — only return the associated or descriptive information related to the '{keyword}' in the files.
                       Return only the exact information related to the '{keyword}' as output without extra content. 
                       Return only the value or descriptive text related to the '{keyword}' — do not include any leading phrases or explanatory text.
                       Return ONLY 'Not Found' if the information related to the '{keyword}' is not present in any files.
                     """
        
        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": enhanced_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        # inputs = inputs.to("cuda")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # Process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        if is_yes_no:
            return "Yes" if result.lower() == "yes" else "No"
        else:
            return result if result.lower() != "not found" else None

    except Exception as e:
        st.error(f"Error in vision extraction: {str(e)}")
        return None

def preprocess_and_extract_batch(batch_images, keyword, excel_row_info):
    """Process a batch of images with its own model instance"""
    # Load model within this process
    model, processor = load_model()
    
    results = []
    is_yes_no = "(Y/N)" in keyword
    all_matches = []  # Track all potential matches
    
    for img_path in batch_images:
        try:
            # Preprocess image
            enhanced_path = preprocess_image(img_path)
            
            # Extract text using the model loaded in this process
            extracted_text = enhanced_extract_with_vision(enhanced_path, keyword, model, processor)
            
            if extracted_text and extracted_text.lower() not in ['not found', 'no']:
                source_file = os.path.basename(img_path).replace("_processed.jpg", "").replace("_page_", " (Page ")
                if "_page_" in os.path.basename(img_path):
                    source_file += ")"
                
                # Calculate confidence score (you can modify this based on your criteria)
                confidence_score = calculate_confidence_score(extracted_text, keyword)
                
                # Track all matches with their confidence
                all_matches.append({
                    "text": extracted_text,
                    "source": source_file,
                    "confidence": confidence_score
                })
                # Remove the break statement to continue searching all files
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # After processing all images, determine the best match
    best_match = find_best_match(all_matches) if all_matches else None
    
    if best_match:
        output_value = "Yes" if is_yes_no else best_match["text"]
        return {
            "Attribute": keyword,
            "Extracted Info": output_value,
            "Source File": best_match["source"],
            "Confidence": best_match["confidence"]
        }
    else:
        return {
            "Attribute": keyword,
            "Extracted Info": "Not Found",
            "Source File": "Not Found",
            "Confidence": "N/A"
        }

def calculate_confidence_score(text, keyword):
    """Calculate a confidence score for the extracted text"""
    # You can implement various heuristics here:
    # - Length of the extracted text (longer might be more complete)
    # - Presence of specific keywords
    # - Text relevance to the attribute
    # - Format of the data (if you expect specific formats)
    
    # Simple implementation - can be enhanced
    if not text:
        return 0
    
    # Base confidence
    confidence = 0.7
    
    # Add confidence for longer responses (assuming more detail)
    if len(text) > 50:
        confidence += 0.1
    
    # Add confidence for presence of the keyword or related terms
    clean_keyword = keyword.replace("(Y/N)", "").strip().lower()
    if clean_keyword in text.lower():
        confidence += 0.1
    
    # You can add more sophisticated checks here
    
    # Convert to human-readable format
    if confidence > 0.9:
        return "Very High"
    elif confidence > 0.7:
        return "High"
    elif confidence > 0.5:
        return "Medium"
    else:
        return "Low"

def find_best_match(matches):
    """Select the best match from all potential matches"""
    if not matches:
        return None
    
    # Define priority order for confidence levels
    confidence_priority = {
        "Very High": 4,
        "High": 3,
        "Medium": 2, 
        "Low": 1
    }
    
    # Sort by confidence score (highest first)
    sorted_matches = sorted(
        matches, 
        key=lambda x: confidence_priority.get(x["confidence"], 0), 
        reverse=True
    )
    
    # Return the highest confidence match
    return sorted_matches[0]

# Add this function to create and maintain worker processes with loaded models
# @st.cache_resource
def create_worker_pool(max_workers=2):
    """Create a pool of worker processes that maintain loaded models"""
    return ProcessPoolExecutor(max_workers=max_workers)

# Modified extraction function that receives a pre-loaded model
def process_with_loaded_model(batch_images, keyword):
    """Process a batch of images with a pre-loaded model for one attribute"""
    # Load model within this process if not already loaded
    if not hasattr(process_with_loaded_model, 'model'):
        torch.cuda.set_device(0)
        process_with_loaded_model.model, process_with_loaded_model.processor = load_model()
        print("Model loaded in worker process")
    
    model = process_with_loaded_model.model
    processor = process_with_loaded_model.processor
    
    is_yes_no = "(Y/N)" in keyword
    found_info = None
    source_file = None
    
    for img_path in batch_images:
        try:
            # Preprocess image
            enhanced_path = preprocess_image(img_path)
            
            # Extract text using the cached model in this process
            extracted_text = enhanced_extract_with_vision(enhanced_path, keyword, model, processor)
            
            if extracted_text and extracted_text.lower() not in ['not found', 'no']:
                found_info = extracted_text
                source_file = os.path.basename(img_path).replace("_processed.jpg", "").replace("_page_", " (Page ")
                if "_page_" in os.path.basename(img_path):
                    source_file += ")"
                # Return early once we find a match
                break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    output_value = "Yes" if (is_yes_no and found_info) else (found_info if found_info else "Not Found")
    
    return {
        "Attribute": keyword,
        "Extracted Info": output_value,
        "Source File": source_file if found_info else "Not Found",
        "Confidence": "High" if found_info else "N/A"
    }

# Modified main extraction function
def extract_data_from_images(excel_file_path, image_paths, max_workers=2):
    """Extract data using persistent worker processes with loaded models"""
    try:
        df = pd.read_excel(excel_file_path)
        if "Attributes" not in df.columns:
            st.error(f"Invalid file format in {excel_file_path}. Expected 'Attributes' column.")
            return None
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None
    
    extracted_data = []
    extraction_progress = st.progress(0, text="Extracting information...")
    
    # Split image paths into batches
    # batch_size = len(image_paths) // max_workers
    # if batch_size < 1:
    #     batch_size = 1
    # image_batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    # st.write(f"Split {len(image_paths)} images into {len(image_batches)} batches")
    
    # total_rows = len(df)

    # Split image paths into batches
    batch_size = len(image_paths) // max_workers
    if batch_size < 1:
        batch_size = 1
    image_batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    
    # Important: Set max_workers equal to the number of batches
    num_batches = len(image_batches)
    max_workers = num_batches  # This ensures one worker per batch
    
    st.write(f"Split {len(image_paths)} images into {num_batches} batches")
    
    total_rows = len(df)

    # Create a persistent executor with worker processes that maintain loaded models
    with create_worker_pool(max_workers=max_workers) as executor:
        # Process each attribute row sequentially
        for index, row in df.iterrows():
            keyword = row["Attributes"].strip()
            extraction_progress.progress(index / total_rows, text=f"Processing attribute: {keyword}")
            
            # Process batches in parallel, with workers that have persistent models
            futures = {executor.submit(process_with_loaded_model, batch, keyword): batch 
                    for batch in image_batches}
            
            batch_results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch_results.append(result)
            
            # Instead of taking the first non-empty result, find the best result across all batches
            all_valid_results = [r for r in batch_results if r["Source File"] != "Not Found"]
            
            if all_valid_results:
                # Find the result with highest confidence
                best_result = max(all_valid_results, key=lambda x: get_confidence_priority(x["Confidence"]))
                extracted_data.append(best_result)
            else:
                # If no batch found anything, add a "Not Found" entry
                extracted_data.append({
                    "Attribute": keyword,
                    "Extracted Info": "Not Found", 
                    "Source File": "Not Found",
                    "Confidence": "N/A"
                })
    
    extraction_progress.progress(1.0, text="Information extraction complete!")
    result_df = pd.DataFrame(extracted_data)
    result_df['Extraction Time'] = pd.Timestamp.now()
    return result_df

# Helper function to prioritize confidence levels
def get_confidence_priority(confidence):
    confidence_priority = {
        "Very High": 4,
        "High": 3,
        "Medium": 2, 
        "Low": 1,
        "N/A": 0
    }
    return confidence_priority.get(confidence, 0)

# Main extraction function
def fetch_and_process_files(excel_file_path, headers):
    """
    Fetches all files from SharePoint sites and OneDrive, then extracts data based on the keywords
    in the input Excel file.
    
    Args:
        excel_file_path: Path to the Excel file containing attributes (keywords).
        headers: Authentication headers for Microsoft Graph API.
    """
    # Load the attributes from Excel file
    try:
        df = pd.read_excel(excel_file_path)
        if "Attributes" not in df.columns:
            st.error(f"Invalid file format in {excel_file_path}. Expected 'Attributes' column.")
            return None
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None
    
    # Create progress indicators
    fetch_progress = st.progress(0, text="Fetching files from SharePoint and OneDrive...")
    
    # Fetch all SharePoint sites
    sites = list_accessible_sites(headers)
    
    # Initialize collection for all files
    all_files = []
    
    # Get OneDrive files
    st.write("Fetching files from OneDrive...")
    onedrive_files = get_onedrive_files(headers)
    all_files.extend(onedrive_files)
    st.write(f"Found {len(onedrive_files)} files in OneDrive")
    
    # Fetch files from all SharePoint sites
    site_count = len(sites)
    for i, (site_id, site_name, _) in enumerate(sites):
        fetch_progress.progress((i / (site_count + 1)), text=f"Fetching from site: {site_name}")
        
        libraries = get_site_libraries(site_id, headers)
        
        for lib in libraries:
            lib_id = lib.get("id")
            lib_name = lib.get("name")
            st.write(f"Fetching from library: {lib_name}")
            
            # Get all files from this library
            site_files = list_all_files_in_library(lib_id, headers)
            all_files.extend(site_files)
    
    fetch_progress.progress(1.0, text="Files fetched successfully!")
    
    if not all_files:
        st.warning("No files were found in any SharePoint site or OneDrive.")
        return None
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download all files
        download_progress = st.progress(0, text="Downloading files...")
        file_count = len(all_files)
        
        temp_file_paths = []
        for i, file_info in enumerate(all_files):
            download_progress.progress(i / file_count, text=f"Downloading {file_info['name']}...")
            
            # Download the file
            file_content = download_file(file_info['drive_id'], file_info['id'], headers)
            if file_content:
                file_path = os.path.join(temp_dir, file_info['name'])
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                temp_file_paths.append(file_path)
        
        download_progress.progress(1.0, text="All files downloaded successfully!")
        
        # Convert files to images
        st.write("Converting files to images for processing...")
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        image_paths = []
        for file_path in temp_file_paths:
            try:
                image_paths.extend(convert_to_image(file_path, images_dir))
            except Exception as e:
                st.error(f"Error converting {os.path.basename(file_path)}: {str(e)}")
        
        if not image_paths:
            st.error("No files could be converted to images for processing.")
            return None
        
        st.write(f"Successfully converted {len(image_paths)} files/pages to images.")
        
        # Process each attribute from the Excel file
        extracted_data = []
        extraction_progress = st.progress(0, text="Extracting information...")
        
        for index, row in df.iterrows():
            keyword = row["Attributes"].strip()
            extraction_progress.progress(index / len(df), text=f"Processing attribute: {keyword}")
            
            is_yes_no = "(Y/N)" in keyword
            found_info = None
            source_file = None
            
            # Search for the keyword in all images
            for i, img_path in enumerate(image_paths):
                file_progress_text = f"Checking file {i+1}/{len(image_paths)} for '{keyword}'"
                st.text(file_progress_text)
                
                extracted_text = enhanced_extract_with_vision(img_path, keyword)
                
                if extracted_text and extracted_text.lower() not in ['not found', 'no']:
                    found_info = extracted_text
                    source_file = os.path.basename(img_path).replace("_processed.jpg", "").replace("_page_", " (Page ")
                    if "_page_" in os.path.basename(img_path):
                        source_file += ")"
                    break  # Stop searching once found
            
            # Process results based on attribute type
            output_value = "Yes" if (is_yes_no and found_info) else (found_info if found_info else "Not Found")
            
            # Store extracted data
            extracted_data.append({
                "Attribute": keyword,
                "Extracted Info": output_value,
                "Source File": source_file if found_info else "Not Found",
                "Confidence": "High" if found_info else "N/A"
            })
        
        extraction_progress.progress(1.0, text="Information extraction complete!")
        
        # Create and return DataFrame with results
        result_df = pd.DataFrame(extracted_data)
        result_df['Extraction Time'] = pd.Timestamp.now()
        
        return result_df

    # Handle authentication redirects
    if 'auth_code' not in st.session_state and 'code' in st.query_params:
        st.session_state.auth_code = st.query_params['code']
        st.rerun()

    if 'auth_url' in st.query_params:
        st.markdown(f'<meta http-equiv="refresh" content="0; url={st.query_params["auth_url"]}">', unsafe_allow_html=True)

def Main():
    # Streamlit UI

    # MDR_FILE_PATH = "MDR.xlsx"
    # IVDR_FILE_PATH = "IVDR.xlsx"

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize session state for part numbers
    if 'part_numbers' not in st.session_state:
        st.session_state.part_numbers = []

    # Authentication UI
    if not st.session_state.authenticated:
        if 'code' in st.query_params:
            auth_code = st.query_params['code']
            headers = get_auth_headers(auth_code)
            if headers:
                st.session_state.headers = headers
                st.session_state.authenticated = True
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Authentication failed. Please try again.")
        else:
            if st.button("Login"):
                st.info("login Initiated")
                auth_url = get_auth_url()
                st.info("Login details obtained")
                st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)

    else:
        # Part number selection section - NEW CODE
        if 'part_numbers' not in st.session_state or not st.session_state.part_numbers:
            st.write("Please enter the part number(s) to process:")
            
            tab1, tab2 = st.tabs(["Single Part Number", "Multiple Part Numbers"])
            
            with tab1:
                part_number = st.text_input("Enter Part Number (Folder Name):", key="single_part")
                if st.button("Set Part Number"):
                    if part_number:
                        st.session_state.part_numbers = [part_number]
                        st.success(f"Part number set: {part_number}")
                        st.rerun()
                    else:
                        st.error("Please enter a valid part number")
            
            with tab2:
                # Add a section to download the template
                st.markdown("### Excel Template")
                st.markdown("Please download and use the following template for uploading part numbers:")
                
                # Create a download button for the template
                template_path = "Part_Numbers_Template.xlsx"  # You'll need to create this file
                with open(template_path, 'rb') as template_file:
                    st.download_button(
                        label="Download Part Numbers Template",
                        data=template_file,
                        file_name="Part_Numbers_Template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download this template to prepare your part numbers for upload"
                    )

                uploaded_file = st.file_uploader("Upload Excel file with part numbers", type=["xlsx", "xls"])
                if uploaded_file is not None:
                    try:
                        part_numbers = read_part_numbers_from_excel(uploaded_file)
                        if part_numbers:
                            if st.button("Process These Part Numbers"):
                                st.session_state.part_numbers = part_numbers
                                st.success(f"Loaded {len(part_numbers)} part numbers")
                                st.rerun()
                        else:
                            st.error("No part numbers found in the file")
                    except Exception as e:
                        st.error(f"Error reading Excel file: {str(e)}")
        
        # Show options for MDR and IVDR only if part numbers are selected
        # elif 'part_numbers' in st.session_state and st.session_state.part_numbers:
        #     st.write(f"Processing part number(s): {', '.join(st.session_state.part_numbers)}")
            
        #     if st.button("Clear Part Numbers", key="clear_parts"):
        #         st.session_state.part_numbers = []
        #         st.rerun()
                
        #     st.write("Select the type of information to extract:")
            
        #     col1, col2 = st.columns(2)

        #     with col1:
        #         if st.button("MDR"):
        #             st.session_state.processing = "MDR"
        #             st.rerun()

        #     with col2:
        #         if st.button("IVDR"):
        #             st.session_state.processing = "IVDR"
        #             st.rerun()

        # # Process files if MDR or IVDR is selected
        # if 'processing' in st.session_state:
        #     selected_file = MDR_FILE_PATH if st.session_state.processing == "MDR" else IVDR_FILE_PATH
        #     st.subheader(f"Extracting {st.session_state.processing} Information")

        #     with st.spinner(f"Processing {st.session_state.processing} attributes..."):

        elif 'part_numbers' in st.session_state and st.session_state.part_numbers:
            st.write(f"Processing part number(s): {', '.join(st.session_state.part_numbers)}")
            
            if st.button("Clear Part Numbers", key="clear_parts"):
                st.session_state.part_numbers = []
                st.rerun()
                
            st.write("Select the type of information to extract:")
            
            col1, col2 = st.columns(2)

            with col1:
                if st.button("MDR"):
                    st.session_state.processing_type = "MDR"
                    st.rerun()

            with col2:
                if st.button("IVDR"):
                    st.session_state.processing_type = "IVDR"
                    st.rerun()

        # Add this section to handle subcategories
        # if 'processing_type' in st.session_state and 'processing' not in st.session_state:
        #     st.subheader(f"Select {st.session_state.processing_type} subcategory:")
            
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         if st.button(f"{st.session_state.processing_type} Device"):
        #             st.session_state.processing = f"{st.session_state.processing_type}_Device"
        #             st.rerun()
            
        #     with col2:
        #         if st.button(f"{st.session_state.processing_type} System"):
        #             st.session_state.processing = f"{st.session_state.processing_type}_System"
        #             st.rerun()
            
        #     if st.button("Back", key="back_to_main"):
        #         st.session_state.pop('processing_type', None)
        #         st.rerun()

        # Add this section to handle subcategories
        if 'processing_type' in st.session_state and 'processing' not in st.session_state:
            st.subheader(f"Select {st.session_state.processing_type} subcategory:")

            if st.session_state.processing_type == "MDR":
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("MDR Device"):
                        st.session_state.processing = "MDR_Device"
                        st.rerun()
                
                with col2:
                    if st.button("MDR System"):
                        st.session_state.processing = "MDR_System"
                        st.rerun()

            elif st.session_state.processing_type == "IVDR":
                # Only show IVDR Device
                if st.button("IVDR Device"):
                    st.session_state.processing = "IVDR_Device"
                    st.rerun()


        # Process files if a subcategory is selected
        if 'processing' in st.session_state:
            # Define file paths based on selected subcategory
            file_paths = {
                "MDR_Device": "MDR device.xlsx",
                "MDR_System": "MDR system.xlsx",
                "IVDR_Device": "IVDR device.xlsx",
                "IVDR_System": "IVDR system.xlsx"
            }
            
            selected_file = file_paths[st.session_state.processing]
            category_name = st.session_state.processing.replace("_", " ")
            
            st.subheader(f"Extracting {category_name} Information")
            
            headers = st.session_state.headers
            part_numbers = st.session_state.part_numbers
            
            # Container to store all Excel files for final ZIP download
            all_excel_files = {}
            
            # Process each part number individually
            for pn_index, part_number in enumerate(part_numbers):
                st.write(f"### Processing Part Number: {part_number} ({pn_index + 1}/{len(part_numbers)})")
                
                with st.spinner(f"Processing {part_number}..."):
                    # **Step 1: Download Files for this specific part number**
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            
                            # OneDrive Downloads for this part number
                            st.info(f"Fetching files from OneDrive for {part_number}...")
                            download_onedrive_files_recursive(None, f"OneDrive/{part_number}/", headers, zip_file, part_number)

                            # SharePoint Downloads for this part number
                            st.info(f"Fetching files from SharePoint for {part_number}...")
                            sites = list_accessible_sites(headers)
                            progress_text = st.empty()

                            for site_id, site_name in sites:
                                libraries = list_document_libraries(site_id, headers)
                                for library in libraries:
                                    drive_id = library.get('id')
                                    lib_name = library.get('name')

                                    matching_folders = search_for_folders_by_name(
                                        drive_id, part_number, headers, progress_text=progress_text
                                    )
                                    for folder in matching_folders:
                                        folder_id = folder["id"]
                                        folder_path = f"SharePoint/{site_name}/{lib_name}{folder['path']}"
                                        st.success(f"Found: {folder_path}")
                                        download_folder_contents(drive_id, folder_id, folder_path, headers, zip_file)

                        # Extract files for this part number
                        zip_buffer.seek(0)
                        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)

                        # **Step 2: Convert Files to Images for this part number**
                        images_dir = os.path.join(temp_dir, 'images')
                        os.makedirs(images_dir, exist_ok=True)

                        image_paths = []
                        for root, _, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                if not file_path.endswith('.zip'):  # Skip zip files
                                    image_paths.extend(convert_to_image(file_path, images_dir))

                        if not image_paths:
                            st.warning(f"No files could be converted to images for part number {part_number}")
                            # Create empty result for this part number
                            result_df = pd.DataFrame({
                                "Attribute": ["No files found"],
                                "Extracted Info": ["Not Found"],
                                "Source File": ["Not Found"],
                                "Confidence": ["N/A"]
                            })
                        else:
                            st.info(f"Successfully converted {len(image_paths)} files/pages to images for {part_number}")

                            # **Step 3: Extract Data using Vision Model for this part number**
                            result_df = extract_data_from_images(selected_file, image_paths)

                        # **Step 4: Add Part Number column and save to Excel**
                        if result_df is not None and not result_df.empty:
                            # Add Part Number column as the first column
                            # result_df.insert(0, 'Part Number', part_number)
                            result_df['Extraction Time'] = pd.Timestamp.now()
                            
                            # Display results for this part number
                            st.write(f"{category_name} Information Extraction Complete for {part_number}!")
                            st.dataframe(result_df)

                            # Save Excel file for this part number
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                result_df.to_excel(writer, index=False, sheet_name='Extracted Info')
                            excel_buffer.seek(0)
                            
                            # Store in dictionary for ZIP creation
                            filename = f"{part_number}_{st.session_state.processing.lower()}_results.xlsx"
                            all_excel_files[filename] = excel_buffer.getvalue()
                            
                            # Individual download button for this part number
                            st.download_button(
                                label=f"Download Results for {part_number}",
                                data=excel_buffer.getvalue(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_{part_number}"
                            )
                        else:
                            st.error(f"Failed to extract information for part number {part_number}")
            
            # **Step 5: Create ZIP file with all Excel files**
            if all_excel_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, excel_data in all_excel_files.items():
                        zip_file.writestr(filename, excel_data)
                
                zip_buffer.seek(0)
                
                # Download all results as ZIP
                st.subheader("Download All Results")
                st.download_button(
                    label=f"Download All {category_name} Results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"all_{st.session_state.processing.lower()}_results.zip",
                    mime="application/zip"
                )
                
                st.success(f"Processing complete! Generated {len(all_excel_files)} Excel files for {len(part_numbers)} part numbers.")
            else:
                st.error("No results were generated for any part numbers.")

            # Clear the processing state
            if st.button("Back"):
                st.session_state.pop('processing', None)
                st.rerun()

if __name__ == "__main__":
    Main()