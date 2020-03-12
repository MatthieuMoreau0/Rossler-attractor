import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from TP import *

class SpeedNet(nn.Module):
    def __init__(self, num_inputs=3, num_outputs = 3):
        super().__init__()
        self.LinearSpeed = nn.Linear(num_inputs,num_outputs)
        self.LinearNewPos = nn.Linear(num_inputs*2,num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.function = torch.nn.LeakyReLU()

    def forward(self, input):
        input = input.view(-1,self.num_inputs)
        speed = self.LinearSpeed(input)
        state = torch.cat([input, speed], dim=1)
        newpos = self.LinearNewPos(state)
        return newpos, speed

class SpeedNN_model():
    def __init__(self, criterion=torch.nn.MSELoss(), lambda_speed=0.01):
        # self.batch_size = batch_size        
        self.criterion = criterion
        self.lambda_speed = lambda_speed
        self.create_model()

    def create_model(self):
        self.model = SpeedNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    def train(self, datasetTrain, batch_size, epochs, shuffle = True):
        train_loader = torch.utils.data.DataLoader(datasetTrain,
        batch_size=batch_size, shuffle=shuffle, num_workers=1)
        for epoch in range(epochs):
            list_loss = []
            for x,y,s in train_loader:
                output, speed = self.model(x)
                loss_out = self.criterion(output, y)
                loss_speed = self.criterion(speed, s)
                loss = loss_out + self.lambda_speed*loss_speed
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                list_loss.append(loss.detach().numpy())
            print(f"Epoch {epoch} : {np.stack(list_loss).mean()}")

    def predict(self, x_eval):
        y_eval = self.model(x_eval)
        return y_eval
    
    def save_weight(self,name_input = "SpeedNN_model.pth"):
        state_dict = self.model.state_dict()
        torch.save(state_dict, name_input)

    def load(self,name_weights='SpeedNN_model.h5'):
        state_dict = torch.load(name_weights)
        self.model.load_state_dict(state_dict)