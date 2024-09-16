import os
import time
from PIL import Image
import numpy as np
import cv2
import requests
import pandas as pd
import concurrent.futures

# Folder structure
save_directory = "processing/images/"
sharpened_directory = "processing/sharpened_images/"
os.makedirs(save_directory, exist_ok=True)
os.makedirs(sharpened_directory, exist_ok=True)

test = pd.read_csv('dataset/test.csv')

def download_image(image_url):
    response = requests.get(image_url)
    img_data = response.content
    file_path = os.path.join(save_directory, image_url.split('/')[-1])
    with open(file_path, 'wb') as handler:
        handler.write(img_data)
    
    return file_path

def sharpen_image(image_url):
    file_path = os.path.join(save_directory, image_url.split('/')[-1])
    img = Image.open(file_path)
    img_cv = np.array(img)

    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    upscaled = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, searchWindowSize=21, templateWindowSize=7)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)
    sharpened_file_path = os.path.join(sharpened_directory, image_url.split('/')[-1])
    cv2.imwrite(sharpened_file_path, sharpened)

def process_image(image_url):
    download_image(image_url)
    sharpen_image(image_url)

def process_images_parallel(image_links):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, image_links)

start_time = time.time()
image_links = test['image_link'].head(20).tolist()
process_images_parallel(image_links)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
