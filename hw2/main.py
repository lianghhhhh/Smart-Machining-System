import json
from test import testModel
from train import trainBinaryModel, trainMultiClassModel

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Test")
    mode = input("Enter mode (1 or 2): ")
    return mode

def loadConfig(configPath):
    with open(configPath, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    mode = selectMode()
    config = loadConfig("config.json")
    if mode == '1':
        trainBinaryModel(config)
        trainMultiClassModel(config)
    elif mode == '2':
        testModel(config)
    else:
        print("Invalid mode selected. Please choose 1 or 2.")
    