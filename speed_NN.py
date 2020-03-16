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
    def __init__(self, num_inputs=3, num_outputs = 3, delta=0.01):
        super().__init__()
        self.LinearSpeed_1 = nn.Linear(num_inputs,200)
        self.LinearSpeed_2 = nn.Linear(200,500)
        self.LinearSpeed_3 = nn.Linear(500,num_outputs)
        # self.LinearNewPos = nn.Linear(503,num_outputs)
        # self.LinearNewPos_2 = nn.Linear(num_inputs*)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.delta=delta
        # self.function = torch.nn.LeakyReLU()

    def forward(self, input):
        input = input.view(-1,self.num_inputs)
        aux = F.elu(self.LinearSpeed_1(input))
        aux = F.elu(self.LinearSpeed_2(aux))
        speed = self.LinearSpeed_3(aux)
        # newpos = input+self.delta*aux
        # aux = torch.cat([F.elu(aux), speed], dim=1)
        newpos = speed
        return newpos, speed

class SpeedNN_model():
    def __init__(self, criterion=torch.nn.SmoothL1Loss(), lambda_speed=0.01):
        # self.batch_size = batch_size        
        self.criterion1 = criterion
        self.lambda_speed = lambda_speed
        self.create_model()

    def create_model(self):
        self.model = SpeedNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)


    def train(self, datasetTrain, batch_size, epochs, shuffle = True, test = None):
        train_loader = torch.utils.data.DataLoader(datasetTrain,
        batch_size=batch_size, shuffle=shuffle, num_workers=1)

        val_loader = torch.utils.data.DataLoader(test,
        batch_size = 10000, num_workers =1)

        for epoch in range(epochs):
            list_loss = []
            list_loss_speed = []
            list_loss_pos = []

            val_list_loss = []
            val_list_loss_speed = []
            val_list_loss_pos = []

            self.model.train()
            for x,y,s in train_loader:
                self.optimizer.zero_grad()
                output, speed = self.model(x)
                loss_out = self.criterion1(output, y)
                # TODO: change this to compute speed with variable delta, here delta = 0.01 hardcoded
                speed=(output-x)/0.01
                loss_speed = self.criterion1(speed, s)
                loss = loss_out + self.lambda_speed*loss_speed
                loss.backward()
                self.optimizer.step()
                list_loss_pos.append(loss_out.detach().numpy())
                list_loss_speed.append(loss_speed.detach().numpy())
                list_loss.append(loss.detach().numpy())
            print(f"Epoch {epoch} : Loss Total {np.stack(list_loss).mean()} Loss Pos {np.mean(list_loss_pos)} Loss Speed {np.mean(list_loss_speed)}")

            self.model.eval()
            for x,y,s in val_loader:
                output, speed = self.model(x)
                loss_out = self.criterion1(output, y)
                speed=(output-x)/0.01
                loss_speed = self.criterion1(speed, s)
                loss = loss_out + self.lambda_speed*loss_speed
                val_list_loss_pos.append(loss_out.detach().numpy())
                val_list_loss_speed.append(loss_speed.detach().numpy())
                val_list_loss.append(loss.detach().numpy())
            print(f"Validation : Loss Total {np.stack(val_list_loss).mean()} Loss Pos {np.mean(val_list_loss_pos)} Loss Speed {np.mean(val_list_loss_speed)}")
            
    def predict(self, x_eval):
        y_eval = self.model(x_eval)
        return y_eval
    
    def save_weight(self,name_input = "SpeedNN_model.pth"):
        state_dict = self.model.state_dict()
        torch.save(state_dict, name_input)

    def load(self,name_weights='SpeedNN_model.pth'):
        state_dict = torch.load(name_weights)
        self.model.load_state_dict(state_dict)
