import torch
from model import CncPredictor
from utils import getData, saveResults

def testModel(config):
    data, scaler = getData(config['data_dir'], mode='test', columns=config['columns'], window_size=config['window_size'])
    
    test_data = torch.tensor(data, dtype=torch.float32) 
    
    data_size = len(config['columns'])
    model = CncPredictor(input_size=data_size, hidden_size=config['hidden_size'], output_size=data_size, num_layers=config['num_layers'], dropout=config['dropout'])
    model.load_state_dict(torch.load(config['model_path']))
    print(f"Model loaded from {config['model_path']}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Predicting next 50 steps...")
    
    predictions = [] 
    
    with torch.no_grad():
        current_input = test_data.to(device) # (N, 10, 45)
        
        for _ in range(50):
            output = model(current_input) # (N, 45)
            
            predictions.append(output.unsqueeze(1)) # (N, 1, 45)
            
            # remove oldest step, add newest prediction
            current_input = torch.cat([current_input[:, 1:, :], output.unsqueeze(1)], dim=1)

    all_preds = torch.cat(predictions, dim=1).cpu().numpy()
    
    # Reshape for Inverse Transform: (N * 50, 45)
    all_preds_flat = all_preds.reshape(-1, all_preds.shape[2])
    
    # Inverse Transform to get real values
    all_preds_real = scaler.inverse_transform(all_preds_flat)
    
    # Save
    saveResults(all_preds_real, config['test_result'], config['columns'])
    print(f"Test results saved to {config['test_result']}")