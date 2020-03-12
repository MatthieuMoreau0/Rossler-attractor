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

nclasses = 20

class dataset(torch.utils.data.Dataset):
    def __init__(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
    
    def __len__(self):
        return len(self.X_train)



class Net(nn.Module):
    def __init__(self, num_inputs=3, num_outputs = 3):
        super().__init__()
        self.Linear1 = nn.Linear(num_inputs,num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.function = torch.nn.LeakyReLU()

    def forward(self, input):
        input = input.view(-1,self.num_inputs)
        output = self.Linear1(input)
        return output



class NN_model():
    def __init__(self, criterion=torch.nn.MSELoss()):
        # self.batch_size = batch_size        
        self.criterion = criterion
        self.create_model()

    def create_model(self):
        self.model = Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    def train(self, datasetTrain, batch_size, epochs, shuffle = True):
        train_loader = torch.utils.data.DataLoader(datasetTrain,
        batch_size=batch_size, shuffle=shuffle, num_workers=1)
        for epoch in range(epochs):
            list_loss = []
            for x,y in train_loader:
                output = self.model(x)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                list_loss.append(loss.detach().numpy())
            print(f"Epoch {epoch} : {np.stack(list_loss).mean()}")

    def predict(self, x_eval):
        y_eval = self.model(x_eval)
        return y_eval
    
    def save_weight(self,name_input = "model.pth"):
        state_dict = self.model.state_dict()
        torch.save(state_dict, name_input)

    def load(self,name_weights='model.h5'):
        state_dict = torch.load(name_weights)
        self.model.load_state_dict(state_dict)

   


def lyapunov_exponent(traj, jacobian, max_it=1000, delta_t=1e-3):

    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        jacob = jacobian(traj[i,:])
        #WARNING this is true for the jacobian of the continuous system!
        w_next = np.dot(expm(jacob * delta_t),w) 
        #if delta_t is small you can use:
        #w_next = np.dot(np.eye(n)+jacob * delta_t,w)
    
        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)
        w = w_next
        if i//(max_it/100)>chk:
            print(i//(max_it/100))
            chk +=1
    
    return  np.mean(np.log(rs), axis=0) / delta_t

def newton(f,jacob,x):
    #newton raphson method
    tol =1
    while tol>1e-5:
        #WARNING this is true for the jacobian of the continuous system!
        tol = x
        x = x-solve(jacob(x),f(v=x))
        tol = norm(tol-x)
    return x
    
     
    
if __name__ == '__main__':

    Niter = 100000
    delta_t = 1e-3
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6,  0.02])
    traj,t = ROSSLER_MAP.full_traj(Niter, INIT)

    x_train = traj[:-1]
    y_train = traj[1:]
    datasetTrain = dataset(x_train.astype("float32"),y_train.astype("float32"))
    model = NN_model()
    model.train(datasetTrain, batch_size=100, epochs=20, shuffle = True)
    model.save_weight()


    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2])
    
    fix_point = newton(ROSSLER_MAP.v_eq,ROSSLER_MAP.jacobian,INIT)

    error = norm(fix_point-ROSSLER_MAP.equilibrium())
    print("equilibrium state :", fix_point, ", error : ", error)
    
    lyap = lyapunov_exponent(traj, ROSSLER_MAP.jacobian, max_it=Niter, delta_t=delta_t)
    print("Lyapunov Exponents :", lyap, "with delta t =", delta_t)

    plt.show()
