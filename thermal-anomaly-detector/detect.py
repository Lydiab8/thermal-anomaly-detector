import numpy as np
import matplotlib.pyplot as plt

def detect_anomaly(model, img):
    recon = model.predict(img[np.newaxis, ..., np.newaxis])[0, ..., 0]
    diff = np.abs(recon - img)
    anomaly_map = diff > 0.2  
    return anomaly_map, recon

def show_anomaly(img, anomaly_map):
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(anomaly_map, cmap='hot')
    plt.title('Anomalie')
    plt.show()
