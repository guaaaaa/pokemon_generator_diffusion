# Pokemon Generation Diffusion Model
A Pokemon generator trained using DDPM

## Dataset
The dataset used for training is located at `data/archive.zip`. It's retrieved from https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images?resource=download and contains 10,437 images in 96x96 resolution from 898 Pokemon in different games

## Model
A pre-trained model using the U-Net architecture with DDPM on 50 epochs is located at `model/pokemon.pth` (this is the state_dict of the model). The full code is in `Diffusion.ipynb`

## Sample outputs
The sample outputs are resized to 64x64. Below are some generated samples

![Figure_2](https://github.com/user-attachments/assets/223c91e3-5636-412f-baa2-8f57bf3b6d8e)

![Figure_3](https://github.com/user-attachments/assets/4cb30f53-1663-4917-be22-1a0076c837d0)

![Figure_1](https://github.com/user-attachments/assets/f772ef04-9755-4d41-91ea-6bf935e27fdb)


## Generate your own Pokemon!
To generate your own Pokemons

1. Clone this repository to your local environment
2. Create a Python virtual environment and install torch and matplotlib in it
3. Open the folder `pokemon_generator_diffusion`
4. Run `make generate`

The generation process should take about 1 to 2 minutes depending on the device. Sometimes the model generations a completely white image, this is because the model can perform poorly occasionally, you can try to generate it again
