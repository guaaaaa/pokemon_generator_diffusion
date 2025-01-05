from unet_model import UNet
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = UNet(device=device)

# For cuda device
# model.load_state_dict(torch.load("model/pokemon.pth"))

# For cpu device
model.load_state_dict(torch.load("model/pokemon.pth", map_location=torch.device("cpu")))

# Parameters
img_size = 64
timestep = 1000
beta = torch.linspace(1e-4, 0.02, timestep).to(device) # Size (1000)
alpha = 1 - beta # Size (1000), each entry represents alpha at each timestep
alpha_hat = torch.cumprod(alpha, dim = 0)

# Generate image
model.eval()
with torch.no_grad():
    x = torch.randn((1, 3, img_size, img_size)).to(device) # Initialize n images (pure noise) with size (n, 3, 64, 64)
    for i in reversed(range(1, timestep)): # Loop from 1000 to 1
        t = (torch.ones(1) * i).long().to(device) # Create timestep of size n (for n images), the timesteps are equal to i value
        predicted_noise = model(x, t) # Predict noise using the model, expect output of size (n, 3, 64, 64)

        # Get alpha, alpha_hat, and beta at this timestep
        alpha_ = alpha[t][:, None, None, None]
        alpha_hat_ = alpha_hat[t][:, None, None, None]
        beta_ = beta[t][:, None, None, None]

        if i > 1:
          noise = torch.randn_like(x) # Sample Gaussian noise
        else:
          noise = torch.zeros_like(x) # At last timestep, no noise

        # 1 step of denoising process
        x = 1 / torch.sqrt(alpha_) * (x - ((1 - alpha_) / torch.sqrt(1 - alpha_hat_)) * predicted_noise) + noise * torch.sqrt(beta_)

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