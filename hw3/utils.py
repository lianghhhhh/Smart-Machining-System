import os
import csv
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def getData(data_dir, mode, columns, window_size):
    if mode == 'train':
        print("Loading training files...")
        folder_path = os.path.join(data_dir, 'Hw3_train')
        name = 'experiment'
        files = sorted(os.listdir(folder_path))
        
        raw_data_list = []
        
        # Load all data first to fit scaler
        for filename in files:
            if name in filename:
                df = pd.read_csv(os.path.join(folder_path, filename))
                raw_data_list.append(df[columns].values)
        
        # Concatenate to fit scaler
        full_array = np.concatenate(raw_data_list, axis=0)
        scaler = MinMaxScaler()
        scaler.fit(full_array)
        joblib.dump(scaler, 'scaler.save') # Save scaler for testing
        print(f"Scaler saved to scaler.save")

        # Process into windows using scaled data
        all_data = []
        labels = []
        plot_data = []
        plot_data_labels = []

        for data_array in raw_data_list:
            # Normalize
            data_scaled = scaler.transform(data_array)
            
            for i in range(len(data_scaled) - window_size):
                all_data.append(data_scaled[i:i+window_size])
                labels.append(data_scaled[i+window_size])
        
        # Get experiment_1.csv file for plotting
        if 'experiment_14.csv' in files:
            df_14 = pd.read_csv(os.path.join(folder_path, 'experiment_14.csv'))
            data_14 = scaler.transform(df_14[columns].values)
            for i in range(len(data_14) - window_size):
                plot_data.append(data_14[i:i+window_size])
                plot_data_labels.append(data_14[i+window_size])

        return np.array(all_data), np.array(labels), np.array(plot_data), np.array(plot_data_labels), scaler

    elif mode == 'test':
        print("Loading test files...")
        folder_path = os.path.join(data_dir, 'Hw3_test_inputs')
        name = 'sample'
        files = sorted(os.listdir(folder_path))
        
        # Load scaler
        if os.path.exists('scaler.save'):
            scaler = joblib.load('scaler.save')
        else:
            raise FileNotFoundError("Scaler not found! Train the model first.")

        all_data = []
        for filename in files:
            if name in filename:
                df = pd.read_csv(os.path.join(folder_path, filename))
                data = df[columns].values
                data_scaled = scaler.transform(data) # Normalize
                
                # Create input window (should be exactly one window per sample file usually)
                if len(data_scaled) >= window_size:
                    # Take the LAST window_size rows to predict the future
                    all_data.append(data_scaled[-window_size:])
                else:
                    print(f"Warning: {filename} is too short!")

        return np.array(all_data), scaler
    
def splitTrainVal(data, labels, val_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * (1 - val_ratio))

    train_data = data[:train_size]
    train_labels = labels[:train_size]
    val_data = data[train_size:]
    val_labels = labels[train_size:]

    return train_data, train_labels, val_data, val_labels

def plotData(plot_data, plot_data_labels, model, device, columns, scaler):
    model.eval()
    
    # Run prediction on the entire sequence of windows
    inputs = torch.tensor(plot_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    
    # Move to CPU and Inverse Transform
    preds_scaled = outputs.cpu().numpy()
    targets_scaled = plot_data_labels
    
    preds_real = scaler.inverse_transform(preds_scaled)
    targets_real = scaler.inverse_transform(targets_scaled)
    
    # plot a faeture in a figure, total 45
    for i in range(45):
        plt.figure(figsize=(10, 5))
        col_name = columns[i]
        plt.plot(targets_real[:, i], label='Actual', alpha=0.7, color='blue')
        plt.plot(preds_real[:, i], label='Predicted', alpha=0.7, color='red')
        plt.title(col_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"exp_14_results/feature_{i+1}.png")
        plt.close()
    
    print("Plots saved to exp_14_results/")

def saveResults(results, filepath, columns):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id'] + columns)
        for i,label in enumerate(results):
            id = (i // 50) + 1 # 50 predictions per sample
            writer.writerow([id] + list(label))