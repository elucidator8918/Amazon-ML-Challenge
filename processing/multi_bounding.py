import os
import time
import numpy as np
import cv2
import requests
import pandas as pd
import concurrent.futures
from PIL import Image
import easyocr
import regex as re

# Folder structure
save_directory = "processing/images/"
bounding_directory = "processing/bounding_images/"
os.makedirs(save_directory, exist_ok=True)
os.makedirs(bounding_directory, exist_ok=True)

test = pd.read_csv('dataset/test.csv')

def download_image(image_url):
    response = requests.get(image_url)
    img_data = response.content
    file_path = os.path.join(save_directory, image_url.split('/')[-1])
    with open(file_path, 'wb') as handler:
        handler.write(img_data)
    return file_path

def sharpen_image(image):
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, h=10, hColor=10, searchWindowSize=21, templateWindowSize=7)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def draw_bounding_boxes_from_url(image_url):
    file_path = os.path.join(save_directory, image_url.split('/')[-1])
    img = Image.open(file_path)

    img_cv = np.array(img)

    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Sharpen the image
    img_cv = sharpen_image(img_cv)

    # Detect and draw bounding boxes (dummy implementation; replace with actual bounding box detection code)
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img_cv)
    number_pattern = re.compile(r'\d')

    for detection in result:
        text = detection[1]
        if number_pattern.search(text):
            top_left = tuple([int(val) for val in detection[0][0]])
            bottom_right = tuple([int(val) for val in detection[0][2]])
            img_cv = cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 1)

    # Save the image with bounding boxes
    output_file_path = os.path.join(bounding_directory, image_url.split('/')[-1])
    cv2.imwrite(output_file_path, img_cv)

def process_image(image_url):
    download_image(image_url)
    draw_bounding_boxes_from_url(image_url)

def process_images_parallel(image_links):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, image_links)

start_time = time.time()
image_links = test['image_link'].head(20).tolist()
process_images_parallel(image_links)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
