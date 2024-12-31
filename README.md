# Latent Diffusion Transformer on MNIST

>**Latent diffusion model** using a Transformer-based backbone trained to learn MNIST distribution. 
<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/e6e4e264-a67c-49cf-b482-3b6f8242f883" alt="mnist">
</div>
<br>

**Autoencoder**: Compresses MNIST images (28x28) into a latent space of dimension 64.  
   - Encoder: Small convolutional network.
   - Decoder: Transpose convolutional layers for image reconstruction.

**Diffusion Process**:
   - Linear schedule for betas across 1000 diffusion steps through DDPM-like process.

**Transformer Backbone**: Acts as the denoising model during the reverse diffusion process.  
   - Grouped query-head attention and feedforward blocks with gelu activations.
