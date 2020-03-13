import numpy as np
import matplotlib.pyplot as plt


def draw_histogram(traj,y):
  labels=["x","y",'z']
  for i in range(3):
    plt.figure()
    coord=y[:,i]
    traj_coord=traj[:,i]

    win_width=np.maximum(np.max(coord),np.max(traj_coord),5)-np.minimum(np.min(coord),np.min(traj_coord))
    nb_bins=50

    coord=coord[np.where(np.abs(coord)<win_width)]
    traj_coord=traj_coord[np.where(np.abs(traj_coord)<win_width)]


    height,x=np.histogram(coord,bins=nb_bins)
    height2,x2=np.histogram(traj_coord,bins=nb_bins)


    plt.bar(x[:-1],height,width=win_width/nb_bins,color='r',alpha=0.3)
    plt.bar(x2[:-1],height2,width=win_width/nb_bins,color='b',alpha=0.3)
    plt.title(f"Histogram for the {labels[i]} coordinate")
    plt.xlabel(f"{labels[i]}")
    plt.ylabel('Occurences')
  plt.show()


if __name__ == '__main__':
    print("Loading... ")
    y=np.loadtxt("y_0.01_smoothl1.dat")
    traj=np.loadtxt("traj_0.01_smoothl1.dat")
    print("DONE")

    print("Drawing histograms..")
    draw_histogram(traj,y)
    print("Done")


#stats intéressantes : max distances entre deux points (au même instant, i.e. erreur de prediction)
#                     - max min distances, les points des deux surfaces les plus éloignés
#                     - 

