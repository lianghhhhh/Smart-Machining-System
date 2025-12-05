import os
import csv
import numpy as np
import pandas as pd

def getData(data_dir, mode, columns, window_size):
    if mode == 'train':
        folder_path = os.path.join(data_dir, 'Hw3_train')
        name = 'experiment'

        files = sorted(os.listdir(folder_path))
        all_data = []
        labels = []
        plot_data = []
        plot_data_labels = []

        for filename in files:
            if name in filename:
                df = pd.read_csv(os.path.join(folder_path, filename))
                data = df[columns].values
                # Create sliding windows for LSTM
                for i in range(len(data) - window_size):
                    all_data.append(data[i:i+window_size])
                    labels.append(data[i+window_size])  # Next step as label
                    if filename == 'experiment_14.csv':  # plot experiment_14 data
                        plot_data.append(data[i:i+window_size])
                        plot_data_labels.append(data[i+window_size])

        all_data = np.array(all_data)
        labels = np.array(labels)
        return all_data, labels, plot_data, plot_data_labels
    elif mode == 'test':
        folder_path = os.path.join(data_dir, 'Hw3_test_inputs')
        name = 'sample'

        files = sorted(os.listdir(folder_path))
        all_data = []

        for filename in files:
            if name in filename:
                df = pd.read_csv(os.path.join(folder_path, filename))
                data = df[columns].values
                # Create sliding windows for LSTM
                for i in range(len(data) - window_size + 1):
                    all_data.append(data[i:i+window_size])

        all_data = np.array(all_data)
        return all_data
    
def splitTrainVal(data, labels, val_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * (1 - val_ratio))

    train_data = data[:train_size]
    train_labels = labels[:train_size]
    val_data = data[train_size:]
    val_labels = labels[train_size:]

    return train_data, train_labels, val_data, val_labels

def saveResults(results, filepath, columns):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id'] + columns)
        for i,label in enumerate(results):
            id = i // 50 + 1 # 50 predictions per sample
            writer.writerow([id] + list(label))