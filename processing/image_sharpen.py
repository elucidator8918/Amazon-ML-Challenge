import pandas as pd
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
import torch
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test = pd.read_csv('../dataset/train.csv')

save_directory = "test/"
os.makedirs(save_directory, exist_ok=True)

def download_image(image_url):
    save_directory = "test/"
    response = requests.get(image_url)
    img_data = response.content
    file_path = os.path.join(save_directory, image_url.split('/')[-1])
    with open(file_path, 'wb') as handler:
        handler.write(img_data)

    return file_path

for i, (image_url, entity_name) in enumerate(zip(test['image_link'].head(20), test['entity_name'].head(20))):
    file_path = download_image(image_url)


def sharpen_image_bounding_box(image):
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, searchWindowSize=21, templateWindowSize=7)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened

def draw_bounding_boxes_from_url(image_url):
    file_path = '/content/test/' + image_url.split('/')[-1]
    print(file_path)
    img = Image.open(file_path)

    img_cv = np.array(img)

    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Sharpen the image
    img_cv = sharpen_image_bounding_box(img_cv)

    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img_cv)
    number_pattern = re.compile(r'\d')

    for detection in result:
        text = detection[1]
        if number_pattern.search(text):
            top_left = tuple([int(val) for val in detection[0][0]])
            bottom_right = tuple([int(val) for val in detection[0][2]])
            img_cv = cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 1)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

for i, (image_url, entity_name) in enumerate(zip(test['image_link'].head(20), test['entity_name'].head(20))):
    print(f"Processing image {i+1}: {image_url}")
    print(f"Entity Name: {entity_name}")
    draw_bounding_boxes_from_url(image_url)
