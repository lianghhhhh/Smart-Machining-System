import torch
from torch import nn
from model import CncPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import getData, splitTrainVal, plotData
from torch.utils.data import TensorDataset, DataLoader

def trainModel(config):
    data, labels, plot_data, plot_data_labels, scaler = getData(config['data_dir'], 'train', config['columns'], config['window_size'])
    
    train_data, train_labels, val_data, val_labels = splitTrainVal(data, labels, val_ratio=0.1)
    
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training model...")
    data_size = len(config['columns'])
    model = CncPredictor(input_size=data_size, hidden_size=config['hidden_size'], output_size=data_size, num_layers=config['num_layers'], dropout=config['dropout'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    writer = SummaryWriter(log_dir=config['log_dir'])

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss} | Val Loss: {val_loss}")
        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Val', val_loss, epoch+1)

    torch.save(model.state_dict(), config['model_path'])
    print(f"Model saved.")
    
    # Plot experiment 14 data
    print("Generating plots for Experiment 14...")
    plotData(plot_data, plot_data_labels, model, device, config['columns'], scaler)
