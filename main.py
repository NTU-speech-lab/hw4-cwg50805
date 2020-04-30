# main.py
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import torch 
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split


import w2v 
from utils import load_training_data, load_testing_data, evaluation
from preprocess import Preprocess
from data import TwitterDataset
from model import LSTM_Net
from train import training
from test import  testing, psuedo_testing

path_prefix = './'
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個 data 的路徑
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
testing_data = os.path.join(path_prefix, 'testing_data.txt')

w2v_path = os.path.join(path_prefix, 'w2v_all.model') # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 40
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 12
lr = 0.0005
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
model_dir = path_prefix # model directory for checkpoint model

print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
train_x, y = load_training_data(train_with_label)
print(type(train_x))
train_x_no_label = load_training_data(train_no_label)

out_nolabel = []
for i in range(3):
    train_x_temp = train_x.copy()
    y_temp = y.copy()
    if i != 0 :
        for j in range(len(out_nolabel)):
            if out_nolabel[j] == 1:
                train_x_temp.append(train_x_no_label[j])
                y_temp.append(1)
            elif out_nolabel[j] == 0:
                train_x_temp.append(train_x_no_label[j])
                y_temp.append(0)
    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x_temp, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_temp = preprocess.sentence_word2idx()
    y_temp = preprocess.labels_to_tensor(y_temp)

    # 製作一個 model 的對象
    #model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=150, num_layers=2, dropout=0.5, fix_embedding=fix_embedding)
    #model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    model = model.to(device)

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_val, X_train, y_val, y_train = train_x_temp[:20000], train_x_temp[180000:], y_temp[:20000], y_temp[180000:]

    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把 data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 8)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)

    # 開始訓練
    training(batch_size, 5, lr, model_dir, train_loader, val_loader, model, device)

    preprocess = Preprocess(train_x_no_label , sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    out_nolabel = psuedo_testing(batch_size, test_loader, model, device)
    print(str(out_nolabel.count(1))+"\n")
    print(str(out_nolabel.count(0))+"\n")


print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)

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

# 以下是使用 command line 上傳到 Kaggle 的方式
# 需要先 pip install kaggle、Create API Token，詳細請看 https://github.com/Kaggle/kaggle-api 以及 https://www.kaggle.com/code1110/how-to-submit-from-google-colab
# kaggle competitions submit [competition-name] -f [csv file path]] -m [message]
# e.g., kaggle competitions submit ml-2020spring-hw4 -f output/predict.csv -m "......"