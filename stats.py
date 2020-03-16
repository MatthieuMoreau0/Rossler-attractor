import numpy as np
import matplotlib.pyplot as plt
from TP import *
from rossler_map import *


def draw_histogram(traj,y):
  labels=["x","y",'z']
  for i in range(3):
    plt.figure()
    coord=y[:,i]
    traj_coord=traj[:,i]

    win_width=np.maximum(np.max(coord),np.max(traj_coord))-np.minimum(np.min(coord),np.min(traj_coord))
    nb_bins=50

    coord=coord[np.where(np.abs(coord)<win_width)]
    traj_coord=traj_coord[np.where(np.abs(traj_coord)<win_width)]


    height,x=np.histogram(coord,bins=nb_bins)
    height2,x2=np.histogram(traj_coord,bins=nb_bins)


    plt.bar(x[:-1],height,width=win_width/nb_bins,color='r',alpha=0.3, label='Simulated')
    plt.bar(x2[:-1],height2,width=win_width/nb_bins,color='b',alpha=0.3, label='True system')
    plt.title(f"Histogram for the {labels[i]} coordinate")
    plt.xlabel(f"{labels[i]}")
    plt.legend()
    plt.ylabel('Occurences')
  plt.show()

def time_correlations(gt_traj, sim_traj, T):
  '''
  Plots the joint distributions of (w(t),w(t+T)) (for each axis) for w in {traj, y}
  '''
  labels = ["x","y","z"]
  for i in range(3):
    sim_coords = sim_traj[:-T,i]
    gt_coords = gt_traj[:-T,i]
    translat_sim_coords = sim_traj[T:,i]
    translat_gt_coords = gt_traj[T:,i]
    
    fig=plt.figure()

    ax1 = fig.add_subplot(1,2,1)
    ax1.hist2d(gt_coords, translat_gt_coords, bins=30)
    plt.ylabel(f"{labels[i]}(t+T)")
    plt.xlabel(f"{labels[i]}(t) (true system)")

    ax2 = fig.add_subplot(1,2,2)
    ax2.hist2d(sim_coords, translat_sim_coords, bins=30)
    plt.xlabel(f"{labels[i]}(t) (simulation)")
    fig.suptitle(f"Time correlation for the {labels[i]} coordinate, T={T}")

  plt.show()


def empiric_lyapounov(traj, max_it, delta_t):
  '''
  Makes no sens so far! Is there really some sense in trying to compute an "empirical Lyapounov"???
  '''
  n = traj.shape[1]
  w = np.eye(n)
  rs = []
  chk = 0

  for i in range(max_it-1):
      # Estimate the jacobian through finite difference
      jacob = (traj[i+1,:]-traj[i,:]) / delta_t # Is this the continuous one?
      #WARNING this is true for the jacobian of the continuous system!
      # w_next = np.dot(expm(jacob * delta_t),w) 
      #if delta_t is small you can use:
      w_next = np.dot(np.eye(n)+jacob * delta_t,w)
  
      w_next, r_next = qr(w_next)

      # qr computation from numpy allows negative values in the diagonal
      # Next three lines to have only positive values in the diagonal
      d = np.diag(np.sign(r_next.diagonal()))
      w_next = np.dot(w_next, d)
      r_next = np.dot(d, r_next.diagonal())

      rs.append(r_next)
      w = w_next
      if i//(max_it/100)>chk:
          #print(i//(max_it/100))
          chk +=1
  
  return  np.mean(np.log(rs), axis=0) / delta_t

def compare_lyapounov(gt_traj, lyap_exp, sim_traj, delta_t):
  '''
  We compare the evolution of the deviation and its expected bounds
  '''
  print(f'Len of the sequence: {sim_traj.shape[0]}')
  time = np.arange(0, sim_traj.shape[0]*delta_t, delta_t)
  print('Time shape:', time.shape)
  deviation_norm = np.linalg.norm((sim_traj-gt_traj), axis=1)
  print('deviation shape:', deviation_norm.shape)
  # Compute e^(lyap*t) for all t
  e_lyap = np.linalg.norm(sim_traj[0]-gt_traj[0]) * np.exp(lyap_exp*np.arange(0, sim_traj.shape[0]*delta_t, delta_t))
  print('e^lyap shape:', e_lyap.shape)

  plt.plot(time, deviation_norm, color='r', label="Simulation deviation")
  plt.plot(time, e_lyap, color='b', label="Lyapounov evolution")
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

if __name__ == '__main__':
    print("Loading... ")
    # y=np.loadtxt("y_0.01_smoothl1.dat")
    y=np.loadtxt("y_0.01.dat")
    # traj=np.loadtxt("traj_0.01_smoothl1.dat")
    traj=np.loadtxt("traj_0.01.dat")
    print("DONE")

    #plot_traj(traj,y)
    
    print("Drawing histograms..")
    draw_histogram(traj,y)
    print("Done")
    
    print("Drawing time correlation distribution..")
    time_correlations(traj, y, 100) # T is chosen at random here, other values should be tested
    print("Done")

    # Right thing to do: compare two generated trajectories (instead of a simulation versus the ground truth)
    print("Computing Lyapounov")
    Niter = 400000
    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    lyap = lyapunov_exponent(traj, ROSSLER_MAP.jacobian, max_it=Niter, delta_t=delta_t)[0]
    print(f'Largest lyapounov coeffiscient of the physical system: {lyap}')
    print("Plotting the deivation vs Lyapounov")
    # Start after 1 to make sure the starting points are different
    compare_lyapounov(traj[10:10000], lyap, y[10:10000], delta_t=delta_t) # We have to focus on the start of the trajectories to avoid overfloat
    print("Done")

    print(('Computing FFT'))
    plot_fourier(traj, y)
    print('Done')

#stats intéressantes : max distances entre deux points (au même instant, i.e. erreur de prediction)
#                     - max min distances, les points des deux surfaces les plus éloignés
#                     - histogrames
#                     - time correlations
#                     - Revisit frequencies

