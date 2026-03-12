import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.drift import DriftNet
from models.diffusion import DiffusionNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained weights
checkpoint = torch.load("outputs/step2_neural_sde_weights.pt", map_location=device)

drift_net = DriftNet().to(device)
diffusion_net = DiffusionNet().to(device)

drift_net.load_state_dict(checkpoint["drift_state_dict"])
diffusion_net.load_state_dict(checkpoint["diffusion_state_dict"])

drift_net.eval()
diffusion_net.eval()

# Load return data
df = pd.read_csv("outputs/nifty50_step1_processed.csv")
returns = torch.tensor(df["logret"].values, dtype=torch.float32).view(-1,1).to(device)

mu_list = []
sigma_list = []

with torch.no_grad():
    for t in range(len(returns)):
        r_t = returns[t].unsqueeze(0)
        t_tensor = torch.tensor([[t/len(returns)]], device=device)

        mu = drift_net(t_tensor, r_t)
        sigma = diffusion_net(t_tensor, r_t)

        mu_list.append(mu.item())
        sigma_list.append(sigma.item())

# Convert to numpy
mu_array = np.array(mu_list)
sigma_array = np.array(sigma_list)

# Save results
np.savez("outputs/step3_mu_sigma.npz", mu=mu_array, sigma=sigma_array)

print("Step 3 complete: μ(t) and σ(t) estimated.")


#plotting(graph)
plt.figure(figsize=(10,4))
plt.plot(mu_array)
plt.title("Estimated Drift μ(t)")
plt.grid(True)
plt.savefig("outputs/mu_plot.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(sigma_array)
plt.title("Estimated Volatility σ(t)")
plt.grid(True)
plt.savefig("outputs/sigma_plot.png", dpi=300)
plt.close()