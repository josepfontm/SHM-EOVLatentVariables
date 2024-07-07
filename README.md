# SHM-EOVLatentVariables

Code to reproduce paper results (or as close as possible, depending on data-availability). 
The scripts implement both latent variable methods and a conventional fully explicit EOV procedure to address environmental and operational variability in vibration-based SHM.

## Code:

* *ETH_Blade_visualization.py*: Plot all PSDs from the dataset between undamaged and damaged conditions.
* *latent_estimation_x.py*: Proposed latent variable approach only considering DSFs (**X**).
* *latent_estimation_x_xi.py*: Proposed latent variable approach considering both DSFs (**X**) and EOPs (**$\xi$**).
* *ols_regression.py*: Explicit EOV procedure which considers EOPs (**$\xi$**) to mitigate the effects observed on DSFs (**X**). This procedure is used to benchmark the results obtained.

## Congress paper:

* [A latent variable approach for mitigation of environmental and operational variability in vibration-based SHM – A linear approach (EWSHM, 2024)](https://www.ndt.net/search/docs.php3?id=29704)
  * Estimation of latent variables (DSFs and EOPs) of structures to obtain a comprehensive baseline model.
  * Avoid common issues with measured EOPs and DSFs: noise-influence, collinearity and/or acting EOPs not acquired during measurement.
  * Application to a [small-scale wind turbine blade](https://doi.org/10.1002/stc.2660).
  * [Presentation of the work at EWSHM2024](https://www.researchgate.net/publication/382052055_A_latent_variable_approach_for_mitigation_of_EOV_in_vibration-based_SHM_-_A_linear_approach).
