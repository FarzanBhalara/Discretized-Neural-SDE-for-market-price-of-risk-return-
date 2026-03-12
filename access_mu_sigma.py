import numpy as np

data = np.load("outputs/step3_mu_sigma.npz")

print(data.files)

mu = data["mu"]
sigma = data["sigma"]

print(mu[:10])
print(sigma[:10])