<<<<<<< HEAD
"""
License Plate Detection and Recognition

Author: Izzatullokh Makhammadjonov
Email: izzatullokhm@gmail.com
GitHub: https://github.com/Izzatullokh24
"""

import streamlit as st
import cv2
import torch
import pytesseract
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO  # Import YOLO from ultralytics
import tempfile
import moviepy.editor as moviepy
import concurrent.futures

# Specify the path to the Tesseract executable (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO('best.pt')  # Load the custom YOLOv8 model

# Function to detect license plates with YOLO and extract text with OCR
def detect_license_plate(frame):
    results = model(frame)  # Perform inference with YOLO
    detected_regions = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get the xyxy format detections

        for box in boxes:
            x_min, y_min, x_max, y_max = box[:4]
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # Extract the detected license plate region
            img_roi = frame[y_min:y_max, x_min:x_max]
            detected_regions.append(img_roi)
            
            # Draw rectangle on the detected license plate
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, 'Number Plate', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detected_regions

# Function to enhance image preprocessing for OCR
def preprocess_for_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply GaussianBlur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Dilate the image to enhance text regions
    kernel = np.ones((1, 1), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    return dilate

# Function to extract text from detected license plates using OCR
def extract_text_from_plates(plates):
    extracted_text = ''
    
    for idx, plate in enumerate(plates):
        processed_plate = preprocess_for_ocr(plate)
        
        # Use Tesseract to extract text from the ROI
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        text = pytesseract.image_to_string(processed_plate, config=custom_config)
        extracted_text += text + '\n\n'  # Add spacing between each extracted text
    
    return extracted_text.strip()

# Function to process a single frame
def process_frame(frame):
    if frame is None:
        return None, ''
    processed_frame, detected_regions = detect_license_plate(frame)
    extracted_text = extract_text_from_plates(detected_regions)
    return processed_frame, extracted_text

# Function to process video frames
def process_video(video_path, skip_frames=1, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file to save the processed video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    extracted_text = ''

    frame_count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_frame = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            future = executor.submit(process_frame, frame)
            future_to_frame[future] = frame

        for future in concurrent.futures.as_completed(future_to_frame):
            processed_frame, text = future.result()
            if processed_frame is not None:
                out.write(processed_frame)
            extracted_text += text + '\n\n'  # Add spacing between each extracted text

    cap.release()
    out.release()

    return temp_video.name, extracted_text.strip()

# Function to handle images
def process_image(image):
    img = np.array(image)  # Convert PIL image to numpy array
    # Detect license plate using YOLOv8
    result_image, detected_regions = detect_license_plate(img)
    extracted_text = extract_text_from_plates(detected_regions)
    
    # Convert result_image to PIL Image for saving
    result_image_pil = Image.fromarray(result_image)
    
    return result_image_pil, extracted_text

# Function to convert an image array to a downloadable link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download Processed Image</a>'
    text_file = f'<a href="data:text/plain;base64,{base64.b64encode(text.encode()).decode()}" download="extracted_text.txt">Download Extracted Text</a>'
    return href, text_file

# Streamlit app
st.title("License Plate Detection and Extract text")


# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

# Display project description
st.markdown("""
## Project Description
This project uses a YOLOv8 model for detecting license plates in images and videos. 
Once the license plates are detected, the text on the plates is extracted using Tesseract OCR. 
You can upload images or videos, and the application will process them to detect license plates 
and extract the text.

**Note:** You can download the processed video which contains the detected license plates highlighted.

## Author Information
**Author:** Izzatullokh Makhammadjonov  
**Email:** izzatullokhm@gmail.com  
**GitHub:** [Izzatullokh24](https://github.com/Izzatullokh24)
""")
if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        st.write("")
        st.write("Processing image...")
        
        # Process the image and extract text
        processed_image, extracted_text = process_image(image)
        
        st.image(processed_image, caption='Processed Image with Detected Plates.', use_column_width=True)
        
        st.write("Extracted Text:")
        st.write(extracted_text)
        
        # Get download links for processed image and extracted text
        img_link, text_link = get_image_download_link(processed_image, "processed_image.jpg", extracted_text)
        
        st.markdown(img_link, unsafe_allow_html=True)
        st.markdown(text_link, unsafe_allow_html=True)
        
    elif file_type == 'video':
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
    
        st.write("")
        st.write("Processing video...")
        
        # Process the video and extract text
        processed_video_path, extracted_text = process_video(video_path, skip_frames=2)
        
        st.write("Processed Video:")
        st.video(processed_video_path)
        
        # Provide a download link for the processed video
        with open(processed_video_path, 'rb') as f:
            bytes_video = f.read()
            b64_video = base64.b64encode(bytes_video).decode()
            href = f'<a href="data:video/mp4;base64,{b64_video}" download="processed_video.mp4">Download Processed Video</a>'
            st.markdown(href, unsafe_allow_html=True)
=======
"""
License Plate Detection and Recognition

Author: Izzatullokh Makhammadjonov
Email: izzatullokhm@gmail.com
GitHub: https://github.com/Izzatullokh24
"""

import streamlit as st
import cv2
import torch
import pytesseract
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO  # Import YOLO from ultralytics
import tempfile
import moviepy.editor as moviepy
import concurrent.futures

# Specify the path to the Tesseract executable (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO('best.pt')  # Load the custom YOLOv8 model

# Function to detect license plates with YOLO and extract text with OCR
def detect_license_plate(frame):
    results = model(frame)  # Perform inference with YOLO
    detected_regions = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get the xyxy format detections

        for box in boxes:
            x_min, y_min, x_max, y_max = box[:4]
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # Extract the detected license plate region
            img_roi = frame[y_min:y_max, x_min:x_max]
            detected_regions.append(img_roi)
            
            # Draw rectangle on the detected license plate
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, 'Number Plate', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detected_regions

# Function to enhance image preprocessing for OCR
def preprocess_for_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply GaussianBlur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Dilate the image to enhance text regions
    kernel = np.ones((1, 1), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    return dilate

# Function to extract text from detected license plates using OCR
def extract_text_from_plates(plates):
    extracted_text = ''
    
    for idx, plate in enumerate(plates):
        processed_plate = preprocess_for_ocr(plate)
        
        # Use Tesseract to extract text from the ROI
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        text = pytesseract.image_to_string(processed_plate, config=custom_config)
        extracted_text += text + '\n\n'  # Add spacing between each extracted text
    
    return extracted_text.strip()

# Function to process a single frame
def process_frame(frame):
    if frame is None:
        return None, ''
    processed_frame, detected_regions = detect_license_plate(frame)
    extracted_text = extract_text_from_plates(detected_regions)
    return processed_frame, extracted_text

# Function to process video frames
def process_video(video_path, skip_frames=1, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file to save the processed video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    extracted_text = ''

    frame_count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_frame = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            future = executor.submit(process_frame, frame)
            future_to_frame[future] = frame

        for future in concurrent.futures.as_completed(future_to_frame):
            processed_frame, text = future.result()
            if processed_frame is not None:
                out.write(processed_frame)
            extracted_text += text + '\n\n'  # Add spacing between each extracted text

    cap.release()
    out.release()

    return temp_video.name, extracted_text.strip()

# Function to handle images
def process_image(image):
    img = np.array(image)  # Convert PIL image to numpy array
    # Detect license plate using YOLOv8
    result_image, detected_regions = detect_license_plate(img)
    extracted_text = extract_text_from_plates(detected_regions)
    
    # Convert result_image to PIL Image for saving
    result_image_pil = Image.fromarray(result_image)
    
    return result_image_pil, extracted_text

# Function to convert an image array to a downloadable link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download Processed Image</a>'
    text_file = f'<a href="data:text/plain;base64,{base64.b64encode(text.encode()).decode()}" download="extracted_text.txt">Download Extracted Text</a>'
    return href, text_file

# Streamlit app
st.title("License Plate Detection and Extract text")


# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

# Display project description
st.markdown("""
## Project Description
This project uses a YOLOv8 model for detecting license plates in images and videos. 
Once the license plates are detected, the text on the plates is extracted using Tesseract OCR. 
You can upload images or videos, and the application will process them to detect license plates 
and extract the text.

**Note:** You can download the processed video which contains the detected license plates highlighted.

## Author Information
**Author:** Izzatullokh Makhammadjonov  
**Email:** izzatullokhm@gmail.com  
**GitHub:** [Izzatullokh24](https://github.com/Izzatullokh24)
""")
if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        st.write("")
        st.write("Processing image...")
        
        # Process the image and extract text
        processed_image, extracted_text = process_image(image)
        
        st.image(processed_image, caption='Processed Image with Detected Plates.', use_column_width=True)
        
        st.write("Extracted Text:")
        st.write(extracted_text)
        
        # Get download links for processed image and extracted text
        img_link, text_link = get_image_download_link(processed_image, "processed_image.jpg", extracted_text)
        
        st.markdown(img_link, unsafe_allow_html=True)
        st.markdown(text_link, unsafe_allow_html=True)
        
    elif file_type == 'video':
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
    
        st.write("")
        st.write("Processing video...")
        
        # Process the video and extract text
        processed_video_path, extracted_text = process_video(video_path, skip_frames=2)
        
        st.write("Processed Video:")
        st.video(processed_video_path)
        
        # Provide a download link for the processed video
        with open(processed_video_path, 'rb') as f:
            bytes_video = f.read()
            b64_video = base64.b64encode(bytes_video).decode()
            href = f'<a href="data:video/mp4;base64,{b64_video}" download="processed_video.mp4">Download Processed Video</a>'
            st.markdown(href, unsafe_allow_html=True)
>>>>>>> d09a5f0a4cb9fbcfdf79a6a8f7537d10fa4d1c87
