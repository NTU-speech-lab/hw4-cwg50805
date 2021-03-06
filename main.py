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
from test import  testing

path_prefix = './'
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 處理好各個 data 的路徑
if (int(sys.argv[1]) == 0):
    train_with_label = sys.argv[2]
    #print(train_with_label)
    train_no_label = sys.argv[3]
elif (int(sys.argv[1]) == 1):
    testing_data = sys.argv[2]
    predict_file = sys.argv[3]



w2v_path = os.path.join(path_prefix, 'w2v_test.model') # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 40
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 12
lr = 0.0005
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
model_dir = path_prefix # model directory for checkpoint model

if int(sys.argv[1]) == 0:
    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個 model 的對象
    model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=150, num_layers=2, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]
    print(X_train[0])

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
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

if int(sys.argv[1]) == 1:
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
    model = torch.load(os.path.join(model_dir, 'ckpt_best.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到 csv 檔案供上傳 Kaggle

    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, predict_file), index=False)
    print("Finish Predicting")
