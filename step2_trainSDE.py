print("HELLO I AM RUNNING")


import torch
import torch.optim as optim
import pandas as pd
torch.manual_seed(0)

from models.drift import DriftNet
from models.diffusion import DiffusionNet
from models.sde_simulator import euler_maruyama_step

# Load processed returns from Step 1
df = pd.read_csv("outputs/nifty50_step1_processed.csv")
returns = torch.tensor(df["logret"].values, dtype=torch.float32).view(-1, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
returns = returns.to(device)

drift_net = DriftNet().to(device)
diffusion_net = DiffusionNet().to(device)

optimizer = optim.Adam(
    list(drift_net.parameters()) + list(diffusion_net.parameters()),
    lr=1e-3
)

dt = 1.0

epochs = 1000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0

    for t in range(len(returns) - 1):
        r_t = returns[t].unsqueeze(0)     # actual log returns from output csv file
        r_tp1 = returns[t + 1].unsqueeze(0) # predicted log returns using euler maryuma

        t_tensor = torch.tensor([[t / len(returns)]], device=device)

        r_hat = euler_maruyama_step(
            r_t,
            t_tensor,
            drift_net,
            diffusion_net,
            dt
        )

        loss += (r_tp1 - r_hat).pow(2).mean()

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 20 ==0:
      print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

print("Training complete.")

#epoch vs loss graph
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.savefig("outputs/loss_plot.png", dpi=300)
plt.close()

print("Loss plot saved to outputs/loss_plot.png")



torch.save({
    "drift_state_dict": drift_net.state_dict(),
    "diffusion_state_dict": diffusion_net.state_dict(),
}, "outputs/step2_neural_sde_weights.pt")
print("Saved to outputs/step2_neural_sde_weights.pt")
