### TODO: Variational AutoEncoder - PatchGAN for moving DDPM into Latent Space (Perceptual Encoder of LDM)

- Loss: L1 reconstruction loss + L2 reconstruction loss + VAE KL loss + PatchGAN Discriminator loss;

- Use 4 Residual Blocks with SiLU activation to construct Encoder and Decoder respectively;

- Try Swin as backbone;

- Only load Decoder for inference.

- Possible References:

[GitHub - znxlwm/pytorch-pix2pix: Pytorch implementation of pix2pix for various datasets.](https://github.com/znxlwm/pytorch-pix2pix)

[深度学习《VAE-GAN》_星海千寻的博客-CSDN博客](https://blog.csdn.net/qq_29367075/article/details/110849112)

[Variational AutoEncoders (VAE) with PyTorch - Alexander Van de Kleut](https://avandekleut.github.io/vae/)
