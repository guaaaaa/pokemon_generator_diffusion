# pokemon_generator_diffusion
A pokemon generator trained using DDPM

## Dataset
The dataset used for training is located at `data/archive.zip`. It's retrieved from https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images?resource=download and contains 10,437 images in 96x96 resolution from 898 Pokemon in different games

## Model
A pre-trained model using the U-Net architecture is located at `model/pokemon.pth` (this is the state_dict of the model)

## Sample outputs
The sample outputs are resized to 64x64

## Generate your own Pokemons!
To generate your own pokemons

1. Clone this repository to your local environment
2. Create a Python virtual environment and install torch and matplotlib in it
3. 
