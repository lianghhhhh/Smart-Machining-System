import os
import torch
from torch import nn
from model import BinaryModel, MultiClassModel
from torch.utils.tensorboard import SummaryWriter
from utils import getData, splitTrainVal, saveResults
from torch.utils.data import TensorDataset, DataLoader

def trainBinaryModel(config):
    print(f"Loading binary data...")
    filenames, data, labels = getData(config['data_dir'], mode='trainBinary')
    train_filenames, train_data, train_labels, val_filenames, val_data, val_labels = splitTrainVal(filenames, data, labels, val_ratio=0.1)
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    train_loader = DataLoader(train_dataset, batch_size=config['binary_model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['binary_model']['batch_size'], shuffle=False)

    print(f"Training binary model...")
    model = BinaryModel()
    model_path = config['binary_model']['model_path']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['binary_model']['learning_rate'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    total_epochs = config['binary_model']['epochs']
    writer = SummaryWriter(log_dir=config['log_dir'])
    val_results = []

    for epoch in range(total_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, targets)
            batch_loss.backward()
            optimizer.step()

            train_acc += (outputs.argmax(dim=1) == targets).sum().item()
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data[0].to(device)
                targets = data[1].to(device)

                outputs = model(inputs)
                batch_loss = loss_fn(outputs, targets)

                val_acc += (outputs.argmax(dim=1) == targets).sum().item()
                val_loss += batch_loss.item()

                if epoch == total_epochs - 1: # Save validation results in the last epoch
                    val_results.extend(outputs.argmax(dim=1).cpu().numpy())

        train_acc /= len(train_dataset)
        train_loss /= len(train_loader)
        val_acc /= len(val_dataset)
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/Binary-Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Binary-Val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/Binary-Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Binary-Val', val_acc, epoch+1)
        writer.flush()

        print(f"Epoch {epoch+1}/{total_epochs} | "
              f"Train Loss: {train_loss}, Train Acc: {train_acc:} | "
              f"Val Loss: {val_loss}, Val Acc: {val_acc}")
        
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'models/binary_{epoch+1}.pth')
            print(f"Model checkpoint saved to models/binary_{epoch+1}.pth")
        
    writer.close()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    saveResults(val_filenames, val_results, config['val_result'], mode='binary')
    print(f"Validation results saved to {config['val_result']}")

def trainMultiClassModel(config):
    print(f"Loading multi-class data...")
    filenames, data, labels = getData(config['data_dir'], mode='trainMulti')
    train_filenames, train_data, train_labels, val_filenames, val_data, val_labels = splitTrainVal(filenames, data, labels, val_ratio=0.1)
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    train_loader = DataLoader(train_dataset, batch_size=config['multi_model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['multi_model']['batch_size'], shuffle=False)

    print(f"Training multi-class model...")
    model = MultiClassModel()
    model_path = config['multi_model']['model_path']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['multi_model']['learning_rate'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    total_epochs = config['multi_model']['epochs']
    writer = SummaryWriter(log_dir=config['log_dir'])
    val_results = []

    for epoch in range(total_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, targets.float())
            batch_loss.backward()
            optimizer.step()

            prob_outputs = torch.sigmoid(outputs)
            predicted = (prob_outputs >= 0.5).float()
            train_acc += (predicted == targets).sum().item() / (targets.size(0) * targets.size(1))
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data[0].to(device)
                targets = data[1].to(device)

                outputs = model(inputs)
                batch_loss = loss_fn(outputs, targets.float())

                prob_outputs = torch.sigmoid(outputs)
                predicted = (prob_outputs >= 0.5).float()
                val_acc += (predicted == targets).sum().item() / (targets.size(0) * targets.size(1))
                val_loss += batch_loss.item()

                if epoch == total_epochs - 1: # Save validation results in the last epoch
                    val_results.extend(predicted.cpu().numpy())
            
        train_acc /= len(train_dataset)
        train_loss /= len(train_loader)
        val_acc /= len(val_dataset)
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/Multi-Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Multi-Val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/Multi-Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Multi-Val', val_acc, epoch+1)
        writer.flush()
        print(f"Epoch {epoch+1}/{total_epochs} | "
              f"Train Loss: {train_loss}, Train Acc: {train_acc:} | "
              f"Val Loss: {val_loss}, Val Acc: {val_acc}")
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'models/multi_{epoch+1}.pth')
            print(f"Model checkpoint saved to models/multi_{epoch+1}.pth")

    writer.close()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    saveResults(val_filenames, val_results, config['val_result'], mode='multi')
    print(f"Validation results saved to {config['val_result']}")