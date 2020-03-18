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
    '''
    3-hidden layers feed-forward neural network
    '''
    def __init__(self, num_inputs=3, num_outputs = 3, delta=0.01):
        super().__init__()
        self.LinearSpeed_1 = nn.Linear(num_inputs,200)
        self.LinearSpeed_2 = nn.Linear(200,500)
        self.LinearSpeed_3 = nn.Linear(500,num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.delta=delta

    def forward(self, input):
        input = input.view(-1,self.num_inputs)
        aux = F.elu(self.LinearSpeed_1(input))
        aux = F.elu(self.LinearSpeed_2(aux))
        newpos = self.LinearSpeed_3(aux)
        return newpos

class SpeedNN_model():
    '''
    Model allowing for speed and jacobian-related losses
    '''
    def __init__(self, criterion=torch.nn.SmoothL1Loss(), lambda_speed=0.01, lambda_jacob=None):
        self.criterion1 = criterion
        self.lambda_speed = lambda_speed
        self.lambda_jacob = lambda_jacob
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

            if self.lambda_jacob is not None:
                list_loss_jac = []
                val_list_loss_jac = []

            self.model.train()
            for x,y,s,j in train_loader:
                if self.lambda_jacob is not None: x.requires_grad = True 

                output = self.model(x)
                loss_out = self.criterion1(output, y)
                loss = loss_out

                if self.lambda_speed is not None:
                    speed=(output-x)/0.01
                    loss_speed = self.criterion1(speed, s)
                    loss += self.lambda_speed*loss_speed

                if self.lambda_jacob is not None:
                    jacobian = torch.zeros((x.shape[0],3,3))
                    for i in range(3): # iterate over each output dimension
                        dim_score = output[:,i].sum()
                        dim_score.backward(retain_graph=True)
                        gradients = x.grad
                        jacobian[:,i,:] = gradients.data
                    loss_jacob = self.criterion1(jacobian, j)
                    loss += self.lambda_jacob * loss_jacob
                    x.requires_grad = False # remove grad from x to avoid modifying the input
                    list_loss_jac.append(loss_jacob.detach().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                list_loss_pos.append(loss_out.detach().numpy())
                list_loss_speed.append(loss_speed.detach().numpy())
                list_loss.append(loss.detach().numpy())

            if self.lambda_jacob is not None:
                print(f"Epoch {epoch} : Loss Total {np.stack(list_loss).mean()} Loss Pos {np.mean(list_loss_pos)} Loss Speed {np.mean(list_loss_speed)} Loss Jacob {np.mean(list_loss_jac)}")
            else:
                print(f"Epoch {epoch} : Loss Total {np.stack(list_loss).mean()} Loss Pos {np.mean(list_loss_pos)} Loss Speed {np.mean(list_loss_speed)}")

            self.model.eval()
            for x,y,s,j in val_loader:
                if self.lambda_jacob is not None: x.requires_grad = True

                output = self.model(x)
                loss_out = self.criterion1(output, y)
                speed=(output-x)/0.01
                loss_speed = self.criterion1(speed, s)
                loss = loss_out + self.lambda_speed*loss_speed

                if self.lambda_jacob is not None:
                    jacobian = torch.zeros((x.shape[0],3,3))
                    for i in range(3): # iterate over each output dimension
                        dim_score = output[:,i].sum()
                        dim_score.backward(retain_graph=True)
                        gradients = x.grad
                        jacobian[:,i,:] = gradients.data
                    loss_jacob = self.criterion1(jacobian, j)
                    loss += self.lambda_jacob * loss_jacob
                    x.requires_grad = False # remove grad from x to avoid modifying the input
                    list_loss_jac.append(loss_jacob.detach().numpy())
                loss.backward() # is it really useful?

                val_list_loss_pos.append(loss_out.detach().numpy())
                val_list_loss_speed.append(loss_speed.detach().numpy())
                val_list_loss.append(loss.detach().numpy())
                
            if self.lambda_jacob is not None:
                print(f"Validation : Loss Total {np.stack(val_list_loss).mean()} Loss Pos {np.mean(val_list_loss_pos)} Loss Speed {np.mean(val_list_loss_speed)} Loss Jacob {np.mean(val_list_loss_jac)}")
            else:
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
