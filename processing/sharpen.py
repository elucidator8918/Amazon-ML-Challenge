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


def sharpen_image(image_url):
    file_path = '/content/test/' + image_url.split('/')[-1]
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
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off') 
    plt.show()


for i, (image_url, entity_name) in enumerate(zip(test['image_link'].head(20), test['entity_name'].head(20))):
    print(f"Processing image {i+1}: {image_url}")
    print(f"Entity Name: {entity_name}")
    sharpen_image(image_url)
