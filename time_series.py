from TP import Net, NN_model

import tqdm
import numpy as np
from numpy.linalg import qr, solve, norm
from scipy.linalg import expm
from rossler_map import RosslerMap

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from speed_NN import *

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10000//self.delta_t)

        self.rosler_nn = SpeedNet() #Net()
        # self.rosler_nn = Net()
        # state_dict = torch.load('model.pth')
        state_dict = torch.load('SpeedNN_model.pth')
        self.rosler_nn.load_state_dict(state_dict)

    def full_traj(self,initial_condition=[-5.75, -1.6,  0.02]): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 
        x = torch.tensor(initial_condition)
        self.rosler_nn.eval()
        with torch.no_grad():
            list_trajectory = []
            for k in tqdm(range(self.nb_steps)):
                y, s = self.rosler_nn(x)
                # y = self.rosler_nn(x)
                # if k%100 == 0 :
                list_trajectory.append(y.detach().numpy().reshape(-1))
                x = y


        return list_trajectory

    def save_traj(self,y,file):
        np.savetxt(file,y)
        #save the trajectory in y.dat file 
    
if __name__ == '__main__':
    delta_t = 1e-2
    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj()
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6,  0.02])
    Niter = ROSSLER.nb_steps

    print(Niter)
    traj,speeds,t = ROSSLER_MAP.full_traj(Niter, INIT)


    # result_loss = np.sum((traj[::100]-y)**2,axis = 1)
    # plt.plot(range(len(result_loss)),result_loss)
    # plt.show()

    print(np.shape(traj))
    print(np.shape(y))

    y = np.stack(y)
    # print(y.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], c = 'b')
    ax.plot(y[:,0], y[:,1], y[:,2], c = 'r')
    plt.show()
    ROSSLER.save_traj(y,f"y_{delta_t}.dat")
    ROSSLER.save_traj(traj,f"traj_{delta_t}.dat")

