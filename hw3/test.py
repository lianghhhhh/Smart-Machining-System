import torch
from model import CncPredictor
from utils import getData, saveResults
from torch.utils.data import TensorDataset, DataLoader

def testModel(config):
    data = getData(config['data_dir'], mode='test', columns=config['columns'], window_size=config['window_size'])
    test_data = torch.tensor(data, dtype=torch.float32) # (num_samples, seq_len, input_size)
    predictions = []

    data_size = len(config['columns'])
    model = CncPredictor(input_size=data_size, hidden_size=config['hidden_size'], output_size=data_size, num_layers=config['num_layers'], dropout=config['dropout'])
    model.load_state_dict(torch.load(config['model_path']))
    print('Testing using model from', config['model_path'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Predict sequentially
    with torch.no_grad():
        for _ in range(50):  # Predict 50 future steps
            inputs = test_data.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            test_data = torch.cat([test_data[:, 1:], outputs.cpu().unsqueeze(1)], dim=1)
    
    saveResults(predictions, config['test_result'], config['columns'])
    print(f"Test results saved to {config['test_result']}")