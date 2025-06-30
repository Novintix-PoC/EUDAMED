import streamlit as st
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
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, set_seed, BitsAndBytesConfig
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
# Clear GPU cache
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model name
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    print("Loading model - this should appear only once")
    # Load model with explicit device handling
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.float16,
    #     device_map="cuda",  # Specify exact device instead of "auto"
    #     attn_implementation="flash_attention_2",
    #     low_cpu_mem_usage=True
    # )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",  # Specify exact device instead of "auto"
        # attn_implementation="flash_attention_2",
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
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # Load processor
    # processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


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
                       Understand the content related to the '{clean_keyword}' in the file and return the correct 'Yes' or 'No' output based on that understanding.
                       Return only 'Yes' or 'No' based on the model's understanding of the content related to '{clean_keyword}' present in the files.
                       Return only 'Not Found' if neither the '{clean_keyword}' nor its related information is mentioned in the files.
                       If the related information for '{clean_keyword}' is present in a table or structured format, the model should understand the content and still return the appropriate 'Yes' or 'No' output.
                       If the input '{clean_keyword}' is represented using a symbol or icon in the input documents, Understand the '{clean_keyword}' that appears in symbol and  return the appropriate 'Yes' or 'No' output based on understanding.
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
                       Understand the symbols in the files and extract the information related to the '{keyword}' based on the understanding of Symbols which represents the '{keyword}'.
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
                       Do not include any extra explanations or context for the output.

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
        inputs = inputs.to("cuda")
        
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

# Modified main extraction function
def extract_data_from_images(excel_file_path, image_paths):
    """Modified version for local file processing"""
    try:
        # Load attributes from Excel
        df = pd.read_excel(excel_file_path)
        if "Attributes" not in df.columns:
            st.error("Excel file must contain 'Attributes' column")
            return None
        
        extracted_data = []
        progress_bar = st.progress(0)
        
        # Process each attribute
        for idx, row in df.iterrows():
            keyword = row["Attributes"].strip()
            progress_bar.progress((idx + 1) / len(df), text=f"Processing: {keyword}")
            
            # Process all images for this attribute
            all_matches = []
            for img_path in image_paths:
                try:
                    enhanced_path = preprocess_image(img_path)
                    extracted_text = enhanced_extract_with_vision(
                        enhanced_path, 
                        keyword,
                        *load_model()  # model and processor
                    )
                    
                    if extracted_text and extracted_text.lower() not in ['not found', 'no']:
                        source = os.path.basename(img_path)
                        source = source.replace("_processed.jpg", "").replace("_page_", " (Page ")
                        if "_page_" in os.path.basename(img_path):
                            source += ")"
                        
                        all_matches.append({
                            "text": extracted_text,
                            "source": source,
                            "confidence": calculate_confidence_score(extracted_text, keyword)
                        })
                except Exception as e:
                    st.error(f"Error processing {img_path}: {str(e)}")
            
            # Select best match
            best_match = find_best_match(all_matches) if all_matches else None
            
            if best_match:
                is_yes_no = "(Y/N)" in keyword
                extracted_data.append({
                    "Attribute": keyword,
                    "Extracted Info": "Yes" if is_yes_no else best_match["text"],
                    "Source File": best_match["source"],
                    "Confidence": best_match["confidence"]
                })
            else:
                extracted_data.append({
                    "Attribute": keyword,
                    "Extracted Info": "Not Found",
                    "Source File": "Not Found",
                    "Confidence": "N/A"
                })
        
        progress_bar.empty()
        result_df = pd.DataFrame(extracted_data)
        result_df['Extraction Time'] = pd.Timestamp.now()
        return result_df
    
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return None

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

def Main():
    # Streamlit UI
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF/DOCX/Images) or a ZIP folder",
        type=["pdf", "docx", "jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    total_size = sum(len(file.getbuffer()) for file in uploaded_files)
    if total_size > 100_000_000:  # 100MB
        st.warning("Large files may take longer to process")

    if uploaded_files:
        # --- MDR/IVDR Selection (Same as Before) ---
        if "processing_type" not in st.session_state:
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

        # --- Subcategory Selection (Same as Before) ---
        if "processing_type" in st.session_state and "processing" not in st.session_state:
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
                if st.button("IVDR Device"):
                    st.session_state.processing = "IVDR_Device"
                    st.rerun()

        # --- Processing Pipeline (Modified for Local Files) ---
        if "processing" in st.session_state:
            # Define Excel template paths
            file_paths = {
                "MDR_Device": "MDR_demo.xlsx",
                "MDR_System": "MDR system.xlsx",
                "IVDR_Device": "IVDR device.xlsx",
                "IVDR_System": "IVDR system.xlsx"
            }
            
            selected_file = file_paths[st.session_state.processing]

            if not os.path.exists(selected_file):
                st.error(f"Template file missing: {selected_file}")
                st.stop() 

            category_name = st.session_state.processing.replace("_", " ")
            
            st.subheader(f"Extracting {category_name} Information")

            with st.spinner("Processing uploaded files..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save all uploaded files to temp_dir
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    
                    # Handle ZIP uploads
                    zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
                    for zip_file in zip_files:
                        zip_path = os.path.join(temp_dir, zip_file.name)
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(temp_dir)

                    # Convert files to images
                    images_dir = os.path.join(temp_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    image_paths = []
                    
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if not file_path.endswith(".zip"):
                                converted = convert_to_image(file_path, images_dir)
                                if converted:
                                    image_paths.extend(converted)

                    if not image_paths:
                        st.error("No valid images could be generated from uploaded files.")
                        return

                    st.success(f"Converted {len(image_paths)} pages to images")

                    # Extract data using AI model
                    result_df = extract_data_from_images(selected_file, image_paths)

                    if result_df is not None:
                        # Display results
                        st.subheader("Extracted Information")
                        st.dataframe(result_df)

                        # Download button
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            result_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="Download Results (Excel)",
                            data=excel_buffer.getvalue(),
                            file_name=f"{st.session_state.processing}_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("Failed to extract information from documents.")

            # Reset button
            if st.button("Start New Extraction"):
                st.session_state.pop("processing", None)
                st.session_state.pop("processing_type", None)
                st.rerun()


if __name__ == "__main__":
    Main()