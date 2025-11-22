import os
import torch
from torch import nn
from model import CnnModel
from torch.utils.data import TensorDataset, DataLoader
from utils import getData, splitTrainVal
from torch.utils.tensorboard import SummaryWriter

def trainModel(config):
    print(f"Training model...")
    data, labels = getData(config['data_dir'], mode='train')
    train_data, train_labels, val_data, val_labels = splitTrainVal(data, labels, val_ratio=0.1)
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CnnModel()
    model_path = config['model_path']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    total_epochs = config['epochs']
    writer = SummaryWriter(log_dir=config['log_dir'])

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