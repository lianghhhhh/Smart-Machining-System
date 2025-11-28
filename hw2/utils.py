import os
import csv
import cv2
import numpy as np

def getData(data_dir, mode):
    if mode == 'trainBinary': # load all data for 0-1 binary classification
        img_path = os.path.join(data_dir, 'public', 'images')
        label_path = os.path.join(data_dir, 'public', 'labels.csv')

        filenames = sorted(os.listdir(img_path))
        data = np.zeros((len(filenames), 1, 256, 256), dtype=np.float32) # image size 256x256, grayscale
        for i, img_name in enumerate(filenames):
            img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
            data[i, 0, :, :] = cv2.resize(img, (256, 256)) # resize to 256x256

        labels = np.zeros((len(filenames)), dtype=np.uint8)
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                label_str = row[1]
                label_indices = [int(x) for x in label_str.split(',')]
                # For binary classification, label as 1 if any defect exists
                if label_indices[0] == 0:
                    labels[i] = 0
                else:
                    labels[i] = 1

        return filenames, data, labels
    
    elif mode == 'trainMulti': # load detected data for multi-class classification
        img_path = os.path.join(data_dir, 'public', 'images')
        label_path = os.path.join(data_dir, 'public', 'labels.csv')

        # Load only images with defects
        labels = []
        defect_filenames = []
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                label_str = row[1]
                label_indices = [int(x) for x in label_str.split(',')]
                label = np.zeros((4,), dtype=np.uint8)
                for j in label_indices:
                    label[j-1] = 1  # shift by 1 for multi-class labels
                if label_indices[0] != 0: # only consider defective samples
                    defect_filenames.append(row[0])
                    labels.append(label)
        labels = np.array(labels)

        filenames = sorted(defect_filenames)
        data = np.zeros((len(filenames), 1, 512, 512), dtype=np.float32) # image size 512x512, grayscale
        for i, img_name in enumerate(filenames):
            img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
            data[i, 0, :, :] = cv2.resize(img, (512, 512)) # resize to 512x512

        return filenames, data, labels

    elif mode == 'test':
        img_path = os.path.join(data_dir, 'private', 'images')

        filenames = sorted(os.listdir(img_path))
        data_1 = np.zeros((len(filenames), 1, 256, 256), dtype=np.float32) # image size 256x256, grayscale
        data_2 = np.zeros((len(filenames), 1, 512, 512), dtype=np.float32) # image size 512x512, grayscale
        for i, img_name in enumerate(filenames):
            img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
            data_1[i, 0, :, :] = cv2.resize(img, (256, 256)) # resize to 256x256
            data_2[i, 0, :, :] = cv2.resize(img, (512, 512)) # resize to 512x512

        return filenames, data_1, data_2
    
def splitTrainVal(filenames, data, labels, val_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * (1 - val_ratio))

    train_filenames = filenames[:train_size]
    train_data = data[:train_size]
    train_labels = labels[:train_size]

    val_filenames = filenames[train_size:]
    val_data = data[train_size:]
    val_labels = labels[train_size:]
    return train_filenames, train_data, train_labels, val_filenames, val_data, val_labels

def saveResults(filenames, results, filepath, mode):
    if mode == 'binary':
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for i, label in enumerate(results):
                writer.writerow([filenames[i], label])
    elif mode == 'multi':
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            binary_filenames = []
            binary_labels = []
            for row in reader:
                binary_filenames.append(row[0])
                binary_labels.append(row[1])

        # if filename exists in binary_filenames, replace its label with multi-class label
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for i, filename in enumerate(binary_filenames):
                if filename in filenames:
                    idx = filenames.index(filename)
                    label_array = results[idx]
                    label_str = ','.join([str(j+1) for j in range(4) if label_array[j] == 1])
                    writer.writerow([filename, label_str])
                else:
                    writer.writerow([filename, binary_labels[i]])
    elif mode == 'test':
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for i, label_array in enumerate(results):
                label_str = ','.join([str(j) for j in range(5) if label_array[j] == 1])
                writer.writerow([filenames[i], label_str])