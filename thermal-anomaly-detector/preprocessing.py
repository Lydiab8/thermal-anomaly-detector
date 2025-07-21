import cv2
import numpy as np

def preprocess_image(path, size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img
