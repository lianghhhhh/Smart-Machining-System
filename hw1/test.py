from model import CnnModel
from utils import getData

def testModel(config):
    data = getData(config['data_dir'], mode='test')
    