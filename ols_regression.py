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

#OLS regression
Fhat = x_m[:,t] @ np.linalg.pinv(xi_m[:,t])   #Calculate regression parameters using t (training data) of input-output data
xhat_ols = Fhat @ xi_m                        #Estimate latent output data given OLS parameter estimation

eta_ols = x_m - xhat_ols   # Error vector between measured DSFs (x_m) and estimated DSFs (x_hat_ols)

SigmaEta_ols = np.cov(eta_ols[:,t])                                    # Covariance matrix
dm2_ols = np.diag(eta_ols.T @ np.linalg.pinv(SigmaEta_ols) @ eta_ols)  # Squared Mahalanobis Distance

SigmaEta_measured = np.cov(x_m[:,t])
dm2_uncorrected = np.diag(x_m.T @ np.linalg.pinv(SigmaEta_measured) @ x_m)

damage_index = 480

training_dm2_ols = dm2_ols[0:400] #Use some validation observations to set damage threshold
damage_threshold = np.percentile(training_dm2_ols, 95)

undamage_ols = dm2_ols[:damage_index] <= damage_threshold
false_alarm_ols = dm2_ols[:damage_index] > damage_threshold
damage_ols = dm2_ols[damage_index:] > damage_threshold
unnoticed_damage_ols = dm2_ols[damage_index:] <= damage_threshold

# Plot dM2
plt.figure()
plt.axhline(damage_threshold, color='black', linestyle='--', linewidth=1, label='95th Percentile Threshold')  # Add threshold line
plt.semilogy(np.arange(damage_index)[undamage_ols], dm2_ols[:damage_index][undamage_ols], '.', color='green', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index)[false_alarm_ols], dm2_ols[:damage_index][false_alarm_ols], '.', color='orange', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index, len(dm2_ols))[unnoticed_damage_ols], dm2_ols[damage_index:][unnoticed_damage_ols], '.', color='blue', markersize=2, alpha=0.5)
plt.semilogy(np.arange(damage_index, len(dm2_ols))[damage_ols], dm2_ols[damage_index:][damage_ols], '.', color='red', markersize=2, alpha=0.5)

plt.axvline(damage_index-0.5, color='black', linewidth=1) #Line induced damage (Scenario A)
plt.axvline(595.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario B)
plt.axvline(715.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario C)
plt.axvline(835.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario D)
plt.axvline(955.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario E)
plt.axvline(1111.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario F)
plt.axvline(1248.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario G)
plt.axvline(1379.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario H)
plt.axvline(1502.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario I)
plt.axvline(1628.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario J)
plt.axvline(1757.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario K)
plt.axvline(1888.5, color='black', linewidth=1,linestyle='--') #Line induced damage (Scenario L)

plt.axvspan(damage_index-0.5,835.5,color='orange',alpha=0.1) #Trainign observations
plt.axvspan(835.5,2015,color='grey',alpha=0.1) #Trainign observations

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



