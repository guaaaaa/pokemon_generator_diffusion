# Pokémon Generation Diffusion Model
A Pokémon generator trained using the **DDPM forward process** and performing the reverse process with both **DDPM** and **stochastic DDIM** sampling.

---

## Dataset

The dataset used for training is located at `data/archive.zip`. It was retrieved from [Kaggle's Pokémon Sprites Dataset](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images?resource=download). 

### Key Details:
- **Size**: 10,437 images
- **Resolution**: 96x96
- **Classes**: 898 Pokémon across various games

---

## Model

The model is a **U-Net** trained with the **DDPM forward process** for 50 epochs. 

### Artifacts:
- **Trained Model**: The trained state_dict is available at `model/pokemon.pth`.
- **Code**: The full implementation is in `Diffusion_updated.ipynb`.

---

## Sample Outputs

The generated samples are resized to 64x64 for display purposes. 

Below are examples of outputs from the model. Empirically, **DDIM** sampling tends to produce better results compared to **DDPM** sampling:

![Sample 1](https://github.com/user-attachments/assets/223c91e3-5636-412f-baa2-8f57bf3b6d8e)

![Sample 2](https://github.com/user-attachments/assets/4cb30f53-1663-4917-be22-1a0076c837d0)

![Sample 3](https://github.com/user-attachments/assets/f772ef04-9755-4d41-91ea-6bf935e27fdb)

---

## Generate Your Own Pokémon!

Follow these steps to generate your own Pokémon:

1. Clone this repository to your local environment
2. Create a Python virtual environment and install torch and matplotlib in it
3. Open the folder by running `cd pokemon_generator_diffusion`
4. Run `make generate`

* Time: The generation process typically takes 1 to 2 minutes, depending on your device.
* Occasional Issues: The model may sometimes produce completely white images due to occasional performance issues. If this happens, simply run the generation process again.

## Potential Next Step
### Current Limitation:
The model was trained with DDPM, so using DDIM with intervals larger than 1 can result in unstable or noisy outputs since the model didn't learn how to handle larger timesteps.

### Future Directions:
* Fine-tuning the model with DDIM: Adapt the model to handle larger intervals for faster sampling.
* Experimenting with stochastic DDIM: Improve stability and performance by tuning the noise schedule (η) or experimenting with hybrid sampling methods.