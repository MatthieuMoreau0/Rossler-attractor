from TP import Net, NN_model


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



class Rossler_model:
    def __int__(self, delta_t):
        self.deta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 10000//self.delta_t

        self.rosler_nn = Net()
        state_dict = torch.load('model.h5')
        self.model.load_state_dict(state_dict)

    def full_traj(self,initial_condition=give_an_initial_condition): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 
        x = torch.tensor(initial_condition)
        list_trajectory = []
        for k in range(self.nb_steps):
            y = self.rosler_nn(x)
            list_trajectory.append(y.detach().numpy())


        return list_trajectory

    def save_traj(self,y):
        #save the trajectory in y.dat file 
    
if __name__ == '__main__':

    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj([-5.75, -1.6,  0.02])
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6,  0.02])
    Niter = ROSSLER.nb_steps
    traj,t = ROSSLER_MAP.full_traj(Niter, INIT)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], c = 'b')
    ax.plot(y[:,0], y[:,1], y[:,2], c = 'r')
    plt.show()
    # ROSSLER.save_traj(y)

