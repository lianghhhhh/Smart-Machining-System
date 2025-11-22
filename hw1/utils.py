import os
import csv
import cv2
import numpy as np

def getData(data_dir, mode):
    label_path = None
    if mode == 'train':
        img_path = os.path.join(data_dir, 'public', 'images')
        label_path = os.path.join(data_dir, 'public', 'labels.csv')
    elif mode == 'test':
        img_path = os.path.join(data_dir, 'private', 'images')

    images = sorted(os.listdir(img_path))
    data = np.zeros((len(images), 1, 300, 300), dtype=np.float32) # image size 300x300, grayscale
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
        data[i, 0, :, :] = cv2.resize(img, (300, 300)) # resize to 300x300

    if label_path:
        labels = np.zeros((len(images),), dtype=np.uint8)
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                labels[i] = int(row[1])
        return data, labels
    else:
        return data
    
def splitTrainVal(data, labels, val_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * (1 - val_ratio))
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    val_data = data[train_size:]
    val_labels = labels[train_size:]
    return train_data, train_labels, val_data, val_labels