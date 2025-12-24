# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

# %%
# Simulate n data points

# Simulate the (X, Y) with correlation rho
# Make a treatment indicator array T
# Add treatment effect to Y_i if T_i = 1

# %%
np.random.seed(67)

# %%
n = 10000 # sample size
tau = 5 # treatment effect

# %%
mean = [0, 0] # mean vector
sd_x, sd_y = 100, 100
rho = 0.6 # correlation
cov_x_y = rho * sd_x * sd_y

cov_matrix = [[sd_x**2, cov_x_y], [cov_x_y, sd_y**2]] # create covariance matrix with correlation rho

x, y = np.random.multivariate_normal(mean, cov_matrix, n).T
# plt.plot(x, y, 'o')
# plt.axis('equal')
# plt.show()


# %%
t = np.repeat([0, 1], n/2) # treatment vector
np.random.shuffle(t)

# %%
y = np.where(t == 1, y + tau, y) # add treatment effect

# %%
# combine data into a dataframe
data = pd.DataFrame({
 'y' : y,
 't' : t,
 'x' : x
})


# %%
treated = data[data['t'] == 1]
control = data[data['t'] == 0]

# Naive estimate
naive_results = ttest_ind(treated.y, control.y, equal_var = True)
ate_naive = np.mean(treated.y) - np.mean(control.y)

print(f"Effect size: {ate_naive:.2f}")
print(f"pvalue : {naive_results.pvalue:.2f}")
print(f"t-statistic : {naive_results.statistic:.2f}")
print(f"std error : {ate_naive / naive_results.statistic:.2f}")

# %%
# CUPED implementation
theta = np.cov(data.x, data.y)[0, 1] / np.var(data.x)
mean_x = np.mean(data.x)
data['y_cv'] = data.y - theta * (data.x - mean_x)

treated = data[data['t'] == 1]
control = data[data['t'] == 0]

# CUPED estimate
cuped_results = ttest_ind(treated.y_cv, control.y_cv, equal_var = True)
ate_cuped = np.mean(treated.y_cv) - np.mean(control.y_cv)

print(f"Effect size: {ate_cuped:.2f}")
print(f"pvalue : {cuped_results.pvalue:.2f}")
print(f"t-statistic : {cuped_results.statistic:.2f}")
print(f"std error : {ate_cuped / cuped_results.statistic:.2f}")

# %%
np.cov(data.x, data.y)[0, 1]

# %%
np.var(data.x)

# %%
plt.scatter(x, y, c=t)
plt.axis('equal')
plt.show()

# %%
