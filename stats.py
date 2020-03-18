import torch
import numpy as np
import matplotlib.pyplot as plt
from TP import *
from rossler_map import *
from dtw import *
from time_series import *


def draw_histogram(gt_traj,sim_traj):
  '''
  Draws the histogram of the trajectories projected along each dimension.
  gt_traj is the sampled trajectory, sim_traj the simulated one.
  '''
  labels=["x","y",'z']
  for i in range(3):
    plt.figure()
    sim_coord = sim_traj[:,i]
    gt_coord = gt_traj[:,i]

    win_width = max(np.max(sim_coord),np.max(gt_coord)) - min(np.min(sim_coord),np.min(gt_coord))
    nb_bins=50

    sim_coord = sim_coord[np.where(np.abs(sim_coord)<win_width)]
    gt_coord=gt_coord[np.where(np.abs(gt_coord)<win_width)]


    height,x=np.histogram(sim_coord,bins=nb_bins)
    height2,x2=np.histogram(gt_coord,bins=nb_bins)


    plt.bar(x[:-1],height,width=win_width/nb_bins,color='r',alpha=0.3, label='Simulated')
    plt.bar(x2[:-1],height2,width=win_width/nb_bins,color='b',alpha=0.3, label='True system')
    plt.title(f"Histogram for the {labels[i]} coordinate")
    plt.xlabel(f"{labels[i]}")
    plt.legend()
    plt.ylabel('Occurences')
  plt.show()

def joint_distrib(gt_traj, sim_traj, T):
  '''
  Plots the joint distributions of (w(t),w(t+T)) (for each axis) for w in {traj, y}.
  gt_traj is the sampled trajectory, sim_traj the simulated one.
  '''
  labels = ["x","y","z"]
  for i in range(3):
    sim_coords = sim_traj[:-T,i]
    gt_coords = gt_traj[:-T,i]
    translat_sim_coords = sim_traj[T:,i]
    translat_gt_coords = gt_traj[T:,i]
    
    fig=plt.figure()

    ax1 = fig.add_subplot(1,2,1)
    ax1.hist2d(gt_coords, translat_gt_coords, bins=20)
    plt.ylabel(f"{labels[i]}(t+T)")
    plt.xlabel(f"{labels[i]}(t) (true system)")

    ax2 = fig.add_subplot(1,2,2)
    ax2.hist2d(sim_coords, translat_sim_coords, bins=20)
    plt.xlabel(f"{labels[i]}(t) (simulation)")
    fig.suptitle(f"Joint distribution for the {labels[i]} coordinate, T={T}")

  plt.show()

def time_correlations(gt_traj, sim_traj, T_list=[5,10,50,100,200,500,1000]):
  '''
  Plots for each dimension the evolution of the correlation between w(t) and w(t=T) as T increases.
  gt_traj is the sampled trajectory, sim_traj the simulated one.
  '''
  labels = ["x", "y", "z"]
  for i in range(3):
    gt_correls = []
    sim_correls = []
    for T in T_list:
      sim_coords = sim_traj[:-T,i]
      gt_coords = gt_traj[:-T,i]
      translat_sim_coords = sim_traj[T:,i]
      translat_gt_coords = gt_traj[T:,i]

      gt_correls.append(np.corrcoef(gt_coords, translat_gt_coords)[0,1])
      sim_correls.append(np.corrcoef(sim_coords, translat_sim_coords)[0,1])

    plt.figure()
    plt.plot(T_list, sim_correls, color='r', label="Simulation correlations")
    plt.plot(T_list, gt_correls, color='b', label="Physical system evolution")
    plt.xlabel('T')
    plt.ylabel('Correlation')
    plt.legend()

  plt.show()


def compare_lyapunov(gt_traj, lyap_exp, sim_traj, delta_t):
  '''
  We compare the evolution of the deviation and its expected bounds.
  gt_traj is the sampled trajectory, sim_traj the simulated one.
  lyap_exp should be the highest lyapunov exponent of the physical system.
  '''
  time = np.arange(0, sim_traj.shape[0]*delta_t, delta_t)
  deviation_norm = np.linalg.norm((sim_traj-gt_traj), axis=1)
  # Compute e^(lyap*t) for all t
  e_lyap = np.linalg.norm(sim_traj[0]-gt_traj[0]) * np.exp(lyap_exp*np.arange(0, sim_traj.shape[0]*delta_t, delta_t))

  plt.plot(time, deviation_norm, color='r', label="Simulation deviation")
  plt.plot(time, e_lyap, color='b', label="Lyapunov evolution")
  plt.legend()
  plt.show()

def plot_fourier(gt_traj, sim_traj):
  '''
  For each dimension individually, compare the Fourier transforms
  '''
  labels=["x","y",'z']
  for i in range(3):
    plt.figure()
    gt_coords = gt_traj[:,i]
    sim_coords = sim_traj[:,i]

    gt_fourier = np.absolute(np.fft.fft(gt_coords))
    sim_fourier = np.absolute(np.fft.fft(sim_coords))


    plt.plot(sim_fourier,color='r', label='Simulated')
    plt.plot(gt_fourier,color='b', label='True system')
    plt.title(f"Fourier transform of the {labels[i]} coordinate")
    plt.legend()
  plt.show()


def plot_traj(gt_traj, sim_traj):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(gt_traj[:,0], gt_traj[:,1], gt_traj[:,2], c = 'b')
  ax.plot(sim_traj[:,0], sim_traj[:,1], sim_traj[:,2], c = 'r')
  plt.show()

def dtws(gt_traj, sim_traj):
  dist_matrix=scipy.spatial.distance_matrix(sim_traj,gt_traj)
  alignment=dtw(dist_matrix, keep_internals=True)

  ## Display the warping curve, i.e. the alignment curve
  alignment.plot(type="alignment")
  a=alignment.index1
  b=alignment.index2
  plt.plot(np.cumsum(alignment.costMatrix[(a,b)]))
  plt.show()



if __name__ == '__main__':
    print("Loading... ")
    y=np.loadtxt("y_0.01_smoothl1.dat")
    traj=np.loadtxt("traj_0.01_smoothl1.dat")
    print("Done")

    plot_traj(traj[:5000],y[:5000])
    
    print("Drawing histograms..")
    draw_histogram(traj,y)
    print("Done")

    print("Plotting joint distribution..")
    joint_distrib(traj, y, T=500)
    print("Done")
    
    print("Drawing time correlation distribution..")
    time_correlations(traj, y, np.arange(10,10000,50))
    print("Done")

    # Using model jacobian to compare equilibrium and Lyapunov exponent
    Niter = 400000
    delta_t = 1e-2

    ROSSLER_model = Rossler_model(delta_t)
    ROSSLER_map = RosslerMap(delta_t=delta_t)

    INIT = np.array([-5.75, -1.6,  0.02])
    fix_point = newton(ROSSLER_map.v_eq,ROSSLER_map.jacobian,INIT)
    jac_at_eq = ROSSLER_model.jacobian(torch.tensor(fix_point).float())
    J = jac_at_eq.copy()
    jac_at_eq[0,0] = 0
    constant = np.array([0,0,ROSSLER_map.b])
    print("Gradient at equilibrium state :", (jac_at_eq-np.eye(3)) @ INIT + constant)
    
    # That computation is very long, execute only if necessary
    '''lyap_gt = lyapunov_exponent(traj, ROSSLER_map.jacobian, max_it=Niter, delta_t=delta_t)
    lyap_sim = lyapunov_exponent(y, ROSSLER_model.jacobian, max_it=Niter, delta_t=delta_t)
    print("True Lyapunov Exponents :", lyap_gt, "with delta t =", delta_t)
    print("Simulation Lyapunov Exponents :", lyap_sim, "with delta t =", delta_t)'''

    print(('Computing FFT'))
    plot_fourier(traj[:4000], y[:4000])
    print('Done')

    print("Computing DTW...")
    dtws(traj[:10000],y[:10000]) #ran on 10000 first steps
    print('Done')
