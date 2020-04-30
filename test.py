# test.py
# 這個 block 用來對 testing_data.txt 做預測
# main.py
import os
import sys
import torch
import argparse
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils import load_training_data, load_testing_data, evaluation
from preprocess import Preprocess
from data import TwitterDataset
from model import LSTM_Net
from train import training


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
            outputs[outputs<0.5] = 0 # 小於 0.5 為負面
            ret_output += outputs.int().tolist()
    
    return ret_output

def psuedo_testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.99] = 1 # 大於等於 0.5 為正面
            outputs[outputs<=0.01] = 0 # 小於 0.5 為負面
            ret_output += outputs.tolist()
    
    return ret_output
'''
# 開始測試模型並做預測
path_prefix = './'
testing_data = os.path.join(path_prefix, 'testing_report.txt')
batch_size = 128
sen_len = 40
w2v_path = os.path.join(path_prefix, 'w2v_all.model') # 處理 word to vec model 的路徑
model_dir = path_prefix # model directory for checkpoint model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading testing data ...")
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle

tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
print("Finish Predicting")
'''