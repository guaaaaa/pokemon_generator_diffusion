from unet_model import UNet
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = UNet(device=device)

# Load state dict
model.load_state_dict(torch.load("model/pokemon.pth", map_location=torch.device(device)))

# Parameters
img_size = 64
timestep = 1000
beta = torch.linspace(1e-4, 0.02, timestep).to(device) # Size (1000)
alpha = 1 - beta # Size (1000), each entry represents alpha at each timestep
alpha_hat = torch.cumprod(alpha, dim = 0)
interval = 1
nu = 0.99
model.eval()
with torch.no_grad():
  x = torch.randn((1, 3, img_size, img_size)).to(device) # Initialize n images (pure noise) with size (n, 3, 64, 64)

  for i in reversed(range(1, timestep, interval)): # Loop from 1000 to 1 with interval of 50
    t = (torch.ones(1) * i).long().to(device) # Create timestep
    predicted_noise = model(x, t) # Predict noise:w
    

    alpha_hat_t = alpha_hat[t][:, None, None, None]
    alpha_hat_t_1 = alpha_hat[t-1][:, None, None, None]
    sigma_t = nu * torch.sqrt((1 - alpha_hat_t_1) / (1 - alpha_hat_t)) * torch.sqrt(1 - (alpha_hat_t / alpha_hat_t_1))

    x = torch.sqrt(alpha_hat_t_1) * ((x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)) + torch.sqrt(1 - alpha_hat_t_1 - torch.square(sigma_t)) * predicted_noise + sigma_t * torch.randn_like(x)

# After reverse process, return x
model.train()
x = (x.clamp(-1, 1) + 1) / 2 # First clamp the pixels to be in the range between -1 and 1, then normalize it to the range between 0 and 1
x = (x * 255).type(torch.uint8)

# Display image
plt.figure(figsize=(7, 7))
plt.imshow(torch.cat([
         torch.cat([i for i in x.cpu()], dim=-1),
     ], dim=-2).permute(1, 2, 0).cpu())
plt.show()
