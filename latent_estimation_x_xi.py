#Load libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

#Load data
t = np.arange(0,200) #Training limit
true_labels = pd.read_csv('data/ETH_Blade/labels.csv', header = None, names = ['measure']) #Load labels damage scenarios
xi_m = np.genfromtxt('data/ETH_Blade/temperature.csv', delimiter=',', dtype=float, autostrip=True) #Load temperature data
xi_m = xi_m.reshape((2015,1))

#Normalize EOPs
scaler = StandardScaler()
xi_m = scaler.fit_transform(xi_m)
xi_m = xi_m.reshape((1,2015))

x_m = pd.read_csv('data/ETH_Blade/psd.csv', header=None)
x_m = x_m.to_numpy()

#Normalize PSDs
x_m = scaler.fit_transform(x_m)

#Swap observations to have undamaged on the front
undamaged_xi_m = xi_m[:,-480:]
damaged_xi_m = xi_m[:,:-480]

undamaged_x_m = x_m[:,-480:]
damaged_x_m = x_m[:,:-480]

#Randomize undamaged observations to have observations from all temperatures in the training and validation sets.
np.random.seed(42)
random_permutation = np.random.permutation(undamaged_x_m.shape[1])
undamaged_xi_m = undamaged_xi_m[:,random_permutation]
undamaged_x_m = undamaged_x_m[:,random_permutation]

xi_m = np.concatenate((undamaged_xi_m,damaged_xi_m),axis=1)
x_m = np.concatenate((undamaged_x_m,damaged_x_m),axis=1)

plt.figure()
plt.scatter(np.arange(0,2015),xi_m[0,:],s=3)
plt.show()

#Regression based on SVD of joint measured data matrix
z_m = np.vstack((x_m, xi_m))                             #Vector of joint input-output data
print(z_m[:,t].shape)
U,S,V = np.linalg.svd(z_m[:,t],full_matrices=False)      #Calculate SVD based on the first 3/4 of data
xhat_svd = U[:4104,:1] @ np.transpose(U[:,:1]) @ z_m     #Estimate latent output data from SVD matrices

eta_svd = x_m - xhat_svd
SigmaEta_svd = np.cov(eta_svd[:,t])

sqrtSigmaEta_svd = U[:4104, 1:4] @ np.diag(S[1:4])
dm2 = np.diag(eta_svd.T @ np.linalg.pinv(SigmaEta_svd) @ eta_svd)   #Squared Mahalanobis Distance

damage_index = 480

training_dm2 = dm2[0:400] #Use some validation observations to set damage threshold
damage_threshold = np.percentile(training_dm2, 95)

undamage = dm2[:damage_index] <= damage_threshold
false_alarm = dm2[:damage_index] > damage_threshold
damage = dm2[damage_index:] > damage_threshold
unnoticed_damage = dm2[damage_index:] <= damage_threshold

# Plot dM2
plt.figure()
plt.axhline(damage_threshold, color='black', linestyle='--', linewidth=1, label='95th Percentile Threshold')  # Add threshold line
#plt.semilogy(dm2, '.',color='navy',markersize=2,alpha=0.5)
plt.semilogy(np.arange(damage_index)[undamage], dm2[:damage_index][undamage], '.', color='green', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index)[false_alarm], dm2[:damage_index][false_alarm], '.', color='orange', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index, len(dm2))[unnoticed_damage], dm2[damage_index:][unnoticed_damage], '.', color='blue', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index, len(dm2))[damage], dm2[damage_index:][damage], '.', color='red', markersize=2, alpha=0.5)
plt.axvline(damage_index, color='black', linewidth=1)
plt.axvspan(0,max(t),color='grey',alpha=0.2)
plt.ylabel('Squared Mahalanobis Distance (SMD)')
plt.xlabel('Observations')
plt.xlim([0,2015])

# Create custom legend
legend_elements = [
    Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=15, label='Undamaged'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='orange', markersize=15, label='False Alarm'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='blue', markersize=15, label='Unnoticed Damage'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='red', markersize=15, label='Damage'),
    Line2D([0], [0], marker='s', color='grey', alpha=0.2, markersize=10,linestyle='None', label='Training Observations'),
]

plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1, 0.5))

plt.show()