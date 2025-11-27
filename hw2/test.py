import torch
from model import BinaryModel, MultiClassModel
from utils import getData, saveResults
from torch.utils.data import TensorDataset, DataLoader

def testModel(config):
    print("Loading test data...")
    filenames, data = getData(config['data_dir'], mode='test')
    test_dataset = TensorDataset(torch.tensor(data))
    test_loader = DataLoader(test_dataset, batch_size=config['binary_model']['batch_size'], shuffle=False)

    binary_model = BinaryModel()
    binary_model.load_state_dict(torch.load(config['binary_model']['model_path']))
    print('Testing using binary model from', config['binary_model']['model_path'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    binary_model.to(device)
    binary_model.eval()
    predictions = []
    with torch.no_grad():
        for test_data in test_loader:
            inputs = test_data[0].to(device)
            outputs = binary_model(inputs)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds.tolist())

    # if binary prediction is 0 (no defect), label as 0
    # if binary prediction is 1 (defect), use multi-class model to predict specific defect types
    final_results = []
    multi_model = MultiClassModel()
    multi_model.load_state_dict(torch.load(config['multi_model']['model_path']))
    print('Testing using multi-class model from', config['multi_model']['model_path'])
    multi_model.to(device)
    multi_model.eval()
    with torch.no_grad():
        for i, pred in enumerate(predictions):
            if pred == 0:
                final_results.append([1, 0, 0, 0, 0])
            else:
                input_dataset = TensorDataset(torch.tensor(data[i].reshape(1, 1, 1600, 256)))
                input_loader = DataLoader(input_dataset, batch_size=config['multi_model']['batch_size'], shuffle=False)
                for batch in input_loader:
                    inputs = batch[0].to(device)
                    outputs = multi_model(inputs)
                    multi_preds = (outputs.sigmoid().cpu().numpy() > 0.5).astype(int)[0].tolist()
                    final_results.append([0] + multi_preds)  # prepend 0 for no-defect class

    saveResults(filenames, final_results, config['test_result'], mode='test')
    print(f"Test results saved to {config['test_result']}")