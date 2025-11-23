import torch
from model import CnnModel
from utils import getData, saveResults
from torch.utils.data import TensorDataset, DataLoader

def testModel(config):
    filenames, data = getData(config['data_dir'], mode='test')
    test_dataset = TensorDataset(torch.tensor(data))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CnnModel()
    model.load_state_dict(torch.load(config['model_path']))
    print('Testing using model from', config['model_path'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            
            prob_outputs = torch.sigmoid(outputs)
            predicted = (prob_outputs >= 0.5).float()
            predictions.extend(predicted.cpu().numpy())
    
    saveResults(filenames, predictions, config['test_result'])
    print(f"Test results saved to {config['test_result']}")