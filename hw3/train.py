import os
import torch
import numpy as np
from torch import nn
from model import CncPredictor
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import getData, splitTrainVal, saveResults
from torch.utils.data import TensorDataset, DataLoader

def trainModel(config):
    print(f"Loading data...")
    data, labels, plot_data, plot_data_labels = getData(config['data_dir'], 'train', config['columns'], config['window_size'])
    train_data, train_labels, val_data, val_labels = splitTrainVal(data, labels, val_ratio=0.1)
    # LSTM expects input shape: (batch, seq_len, input_size)
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training model...")
    data_size = len(config['columns'])
    model = CncPredictor(input_size=data_size, hidden_size=config['hidden_size'], output_size=data_size, num_layers=config['num_layers'], dropout=config['dropout'])
    model_path = config['model_path']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    total_epochs = config['epochs']
    writer = SummaryWriter(log_dir=config['log_dir'])
    val_results = []

    for epoch in range(total_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # inputs: (batch, seq_len, input_size)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, targets)
            batch_loss.backward()
            optimizer.step()
            # For regression accuracy: count samples where all columns' prediction error < 5%
            preds = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            # Calculate relative error per column
            error = np.abs(preds - targets_np) / (np.abs(targets_np) + 1e-8)
            correct = (error < 0.05).all(axis=1).sum()
            train_acc += correct
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                batch_loss = loss_fn(outputs, targets)

                preds = outputs.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()
                error = np.abs(preds - targets_np) / (np.abs(targets_np) + 1e-8)
                correct = (error < 0.05).all(axis=1).sum()
                val_acc += correct
                val_loss += batch_loss.item()

                if epoch == total_epochs - 1:  # Save validation results in the last epoch
                    val_results.extend(preds)

        train_acc /= len(train_dataset)
        train_loss /= len(train_loader)
        val_acc /= len(val_dataset)
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Val', val_acc, epoch+1)
        writer.flush()

        print(f"Epoch {epoch+1}/{total_epochs} | "
              f"Train Loss: {train_loss}, Train Acc: {train_acc:} | "
              f"Val Loss: {val_loss}, Val Acc: {val_acc}")
        
    writer.close()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plotData(plot_data, plot_data_labels, model, device, config)

def plotData(plot_data, plot_data_labels, model, device, config):
    model.eval()
    with torch.no_grad():
        for i in range(len(plot_data)):
            input_seq = torch.tensor(plot_data[i], dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, input_size)
            output = model(input_seq)
            predictions = output.cpu().numpy().squeeze(0)  # (output_size,)

            plt.figure(figsize=(12, 6))
            for j, col in enumerate(config['columns']):
                plt.subplot(len(config['columns'])//3 + 1, 3, j+1)
                plt.plot(range(len(plot_data[i])), plot_data[i][:, j], label='Input Sequence')
                plt.scatter(len(plot_data[i]), plot_data_labels[i][j], color='green', label='True Next Step')
                plt.scatter(len(plot_data[i]), predictions[j], color='red', label='Predicted Next Step')
                plt.title(f'Column: {col}')
                plt.legend()
            plt.tight_layout()
            plt.savefig("experiment_14.png")
            plt.close()