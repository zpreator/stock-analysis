from Model import TrainStonksModel, PredictStonks
import os
import numpy as np

date ='2020-08-18'
repo_dir = os.path.dirname(__file__) # Gets the directory the repo is in
stocks_list = np.loadtxt(os.path.join(repo_dir, 'stocks.txt'), dtype=str)

for stonk in stocks_list:
    model_path = os.path.join(repo_dir, 'models', stonk + '.h5')
    if os.path.exists(model_path):
        PredictStonks(stonk, model_path, date)
    else:
        TrainStonksModel(stonk, model_path)
        PredictStonks(stonk, model_path, date)
