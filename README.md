

# StableSVC: Latent Diffusion Model for Singing Voice Conversion

**TODO:**

- [x] Debug cross-attention module;
- [x] Make dimensionality reduction adjustable in UNet;
- [x] Debug Whisper Embedding module;
- [ ] Implement Classifier-Free Guidance (CFG);
- [ ] Implement VAE-PatchGAN;
- [ ] Implement LDM.

Propose **Latent Diffusion Model (LDM)**[1] for Singing Voice Conversion (SVC)

![image](https://github.com/SLPcourse/MDS6002-222041038-JiahaoChen/blob/main/StableSVC.jpg)

**Current Implementation:**

Simple diffusion for SVC (Denosing Diffusion Probabilistic Model, DDPM[2])

![image](https://github.com/AxiumCrisis61/StableSVC/blob/main/simple_diffusion.jpg)

Whisper CNN module for processing Whisper embedding:

![avatar](https://github.com/AxiumCrisis61/StableSVC/blob/main/Whisper%20CNN.jpg)

**Tentative Results** (see /demo for audios)

![image](https://github.com/AxiumCrisis61/StableSVC/blob/main/denoising_process.jpg)

**See Report for more details**

![image](https://github.com/AxiumCrisis61/StableSVC/blob/main/poster_1.jpg)

![image](https://github.com/AxiumCrisis61/StableSVC/blob/main/poster_2.jpg)

**Link for Google Drive working directory:**
https://drive.google.com/drive/folders/1hY9YPVmqGFB9UIN0WWdJQCAfGAP9-G-1?usp=sharing

**References:**

[1] Rombach R, Blattmann A, Lorenz D, et al. High-resolution image synthesis with latent diffusion models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10684-10695.

[2] Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.

[3] Liu S, Cao Y, Su D, et al. Diffsvc: A diffusion probabilistic model for singing voice conversion[C]//2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021: 741-748.

[4] Liu H, Chen Z, Yuan Y, et al. Audioldm: Text-to-audio generation with latent diffusion models[J]. arXiv preprint arXiv:2301.12503, 2023.

[5] Wang Y, Ju Z, Tan X, et al. AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models[J]. arXiv preprint arXiv:2304.00830, 2023.
