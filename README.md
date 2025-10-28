# Synthetic-Anime-Face-Generation-Using-Denoising-Diffusion-Probabilistic-Models
A Diffusion Approach for Cartoon Face Generation
<img width="676" height="679" alt="image" src="https://github.com/user-attachments/assets/037804b5-2354-4c9a-9d8b-42fe531783fb" />

# Abstract
We present a diffusion-based approach for generating anime-style faces using Denoising Diffusion Probabilistic Models (DDPMs). Our method trains a U-Net architecture on an animation face dataset to iteratively transform random noise into coherent anime portraits. The model produces diverse facial features, hairstyles, and expressions while maintaining the distinctive aesthetic of anime art. Experimental results demonstrate stable training and high-quality generation of 100 unique anime faces.

# Steps
- download the anime face dataset on the Kaggle and put it under the dataset folder
- create a virtual environment
- pip install -r requirement.txt
- run the train.py (it will output 4 images after each epoch for your reference, and save the current model under model folder)
- training might take a whole day
- choose the best model you would like to use, and run the generate.py
