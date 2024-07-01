#Load libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, auc
from matplotlib.ticker import MaxNLocator

#Load data
true_labels = pd.read_csv('data/ETH_Blade/labels.csv', header = None, names = ['measure']) #Load labels damage scenarios
xi_m = pd.read_csv('data/ETH_Blade/temperature.csv', header = None, names = ['temperature']) #Load temperature data
xi_m = xi_m.to_numpy()
xi_m = xi_m.T

x_m = pd.read_csv('data/ETH_Blade/psd.csv', header=None)
x_m = x_m.to_numpy()

#Swap observations to have undamaged on the front
undamaged_x_m = x_m[:,-480:]
damaged_x_m = x_m[:,:-480]

x_m = np.concatenate((undamaged_x_m,damaged_x_m),axis=1)

t = np.arange(0,380) #Training limit
f = np.linspace(0,400,513)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 18))

#Undamaged
ax1.semilogy(f,x_m[513*0:513*1,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 1
ax1.semilogy(f,x_m[513*1:513*2,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 2
ax1.semilogy(f,x_m[513*2:513*3,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 3
ax1.semilogy(f,x_m[513*3:513*4,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 4
ax1.semilogy(f,x_m[513*4:513*5,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 5
ax1.semilogy(f,x_m[513*5:513*6,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 6
ax1.semilogy(f,x_m[513*6:513*7,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 7
ax1.semilogy(f,x_m[513*7:513*8,0:480],color='green',linewidth = 0.1,alpha=0.05) #Acc 8

ax1.set_xlim([0,400])
ax1.set_ylim([1e-5,10])
ax1.set_ylabel('Power Spectral Density (PSD)')
ax1.legend(['Undamaged'],loc='upper right')

#Damaged
ax2.semilogy(f,x_m[513*0:513*1,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 1
ax2.semilogy(f,x_m[513*1:513*2,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 2
ax2.semilogy(f,x_m[513*2:513*3,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 3
ax2.semilogy(f,x_m[513*3:513*4,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 4
ax2.semilogy(f,x_m[513*4:513*5,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 5
ax2.semilogy(f,x_m[513*5:513*6,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 6
ax2.semilogy(f,x_m[513*6:513*7,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 7
ax2.semilogy(f,x_m[513*7:513*8,480:],color='red',linewidth = 0.1,alpha=0.05) #Acc 8

ax2.set_xlim([0,400])
ax2.set_ylim([1e-5,10])
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Power Spectral Density (PSD)')

plt.show()