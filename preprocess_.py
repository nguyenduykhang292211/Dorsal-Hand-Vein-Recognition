
import numpy as np
import os
import cv2
from PIL import Image


def enhance_image(image):
    enhanced_image = cv2.equalizeHist(image)
    return enhanced_image

def remove_noise(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def local_binarization(image):
    block_size = 35
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 10)
    return binary_image

def process(image):
    enhanced = enhance_image(image)
    denoised = remove_noise(enhanced)
    binary = local_binarization(denoised)
    return binary

def preprocess_image(input_path, output_path):
    image = Image.open(input_path)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = process(image)
    return processed_image
    cv2.imwrite(output_path, processed_image)

def preprocess_images_in_directory(input_directory, output_directory):
    if not os.path.exists(input_directory):
        print(f"Thư mục {input_directory} không tồn tại.")
        return

    os.makedirs(output_directory, exist_ok=True)

    for root, dirs, files in os.walk(input_directory):
        for dir_name in dirs:
            input_subdirectory = os.path.join(root, dir_name)
            output_subdirectory = os.path.join(output_directory, dir_name)

            os.makedirs(output_subdirectory, exist_ok=True)
            for file_name in os.listdir(input_subdirectory):
                input_path = os.path.join(input_subdirectory, file_name)
                output_path = os.path.join(output_subdirectory, file_name)

                preprocess_image(input_path, output_path)
