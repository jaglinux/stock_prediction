# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:43:00 2020

@author: jkrishna
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def prepare_data(data_raw, slide):
    data = []
    data_raw = data_raw.to_numpy()
    #print(data_raw)
    for i in range(len(data_raw)-slide):
        data.append(data_raw[i:i+slide])
    data = np.array(data)
    print(data)
    print(data.shape)
    # train data 80 % test data 20 %
    test_data_size = int(np.round(0.2*data.shape[0]))
    print(test_data_size)
    train_data_size = data.shape[0] - test_data_size
    print(train_data_size)
    train_x = data[:train_data_size, :-1, :]
    train_y = data[:train_data_size, -1, :]
    test_x = data[train_data_size:, :-1, :]
    test_y = data[train_data_size:, -1, :]
    #print('Train data is')
    #print(train_x, train_y)
    #print('Test data is')
    #print(test_x, test_y)
    return (train_x, train_y, test_x, test_y)

data = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv')
data = data.sort_values('Date')
#print(data.head())
#print(data.tail())
#print(data.info())

close = data[['Close']]
#print(close)
#print(close.info())

scale = MinMaxScaler(feature_range=(-1,1))
close['Close'] = scale.fit_transform(close['Close'].values.reshape(-1, 1))
#print(close)

train_x, train_y, test_x, test_y = prepare_data(close, slide=20)

# create tensors
train_x = torch.from_numpy(train_x).type(torch.Tensor)
train_y = torch.from_numpy(train_y).type(torch.Tensor)
test_x = torch.from_numpy(test_x).type(torch.Tensor)
test_y = torch.from_numpy(test_y).type(torch.Tensor)

class lstm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers):
        super(lstm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        out,_ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1
output_dim = 1
hidden_dim = 32
layers = 2
epochs = 100
lr = 0.01
 
model = lstm(input_dim, output_dim, hidden_dim, layers)
loss = torch.nn.MSELoss(reduce='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(epochs):
    y_hat = model(train_x)
    epoch_loss = loss(y_hat, train_y)
    print('for epoch', i, 'loss is', epoch_loss.item())
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()

#prediction time !!!
test_y_hat = model(test_x)
test_y_hat = scale.inverse_transform(test_y_hat.detach().numpy())
test_y = scale.inverse_transform(test_y.detach().numpy())
print('Prediction:', test_y_hat[:10])
print('Actual_values: ', test_y[:10])
print('Prediction:', test_y_hat[-10:])
print('Actual_values: ', test_y[-10:])
