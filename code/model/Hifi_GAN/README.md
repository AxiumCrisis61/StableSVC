Borrowed from: https://github.com/jik876/hifi-gan

Adjustments to Configuration v1 used in this project (with last four following [2]):

- Sampling Rate: 22050 to 16000;

- Hop size: 256 to 240;

- Upsample Rate: 8 × 8 × 2 × 2 to 8 × 5 × 3 × 2;

- Upsample Kernel Sizes: [16,16,4,4] to [15, 15, 5, 5];

- Modify up-sampling block: decompose 1-d Deconvolution to Resize-Convolution to avoid Checkerboard Artifacts [3]. 
  
  - Small Notes: number of channels in Conv1d is initial the number of mel-frequency bins, that is to say, channel dimension in the model is essentially the 

# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

### Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

In our [paper](https://arxiv.org/abs/2010.05646), 
we proposed HiFi-GAN: a GAN-based model capable of generating high fidelity speech efficiently.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. 
Although such methods improve the sampling efficiency and memory usage, 
their sample quality has not yet reached that of autoregressive and flow-based generative models. 
In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. 
As speech audio consists of sinusoidal signals with various periods, 
we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. 
A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method 
demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than 
real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen 
speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times 
faster than real-time on CPU with comparable quality to an autoregressive counterpart.

Visit our [demo website](https://jik876.github.io/hifi-gan-demo/) for audio samples.

## Pre-requisites

1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
   And move all wav files to `LJSpeech-1.1/wavs`

## Training

```
python train.py --config config_v1.json
```

To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

Validation loss during training with V1 generator.<br>
![validation loss](./validation_loss.png)

## Pretrained Model

You can also use pretrained models we provide.<br/>
[Download pretrained models](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)<br/> 
Details of each folder are as in follows:

| Folder Name  | Generator | Dataset   | Fine-Tuned                                             |
| ------------ | --------- | --------- | ------------------------------------------------------ |
| LJ_V1        | V1        | LJSpeech  | No                                                     |
| LJ_V2        | V2        | LJSpeech  | No                                                     |
| LJ_V3        | V3        | LJSpeech  | No                                                     |
| LJ_FT_T2_V1  | V1        | LJSpeech  | Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| LJ_FT_T2_V2  | V2        | LJSpeech  | Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| LJ_FT_T2_V3  | V3        | LJSpeech  | Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2)) |
| VCTK_V1      | V1        | VCTK      | No                                                     |
| VCTK_V2      | V2        | VCTK      | No                                                     |
| VCTK_V3      | V3        | VCTK      | No                                                     |
| UNIVERSAL_V1 | V1        | Universal | No                                                     |

We provide the universal model with discriminator weights that can be used as a base for transfer learning to other datasets.

## Fine-Tuning

1. Generate mel-spectrograms in numpy format using [Tacotron2](https://github.com/NVIDIA/tacotron2) with teacher-forcing.<br/>
   The file name of the generated mel-spectrogram should match the audio file and the extension should be `.npy`.<br/>
   Example:
   
   ```
   Audio File : LJ001-0001.wav
   Mel-Spectrogram File : LJ001-0001.npy
   ```

2. Create `ft_dataset` folder and copy the generated mel-spectrogram files into it.<br/>

3. Run the following command.
   
   ```
   python train.py --fine_tuning True --config config_v1.json
   ```
   
    For other command line options, please refer to the training section.

## Inference from wav file

1. Make `test_files` directory and copy wav files into the directory.

2. Run the following command.
   
   ```
   python inference.py --checkpoint_file [generator checkpoint file path]
   ```
   
   Generated wav files are saved in `generated_files` by default.<br>
   You can change the path by adding `--output_dir` option.

## Inference for end-to-end speech synthesis

1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
   You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
   [Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.

2. Run the following command.
   
   ```
   python inference_e2e.py --checkpoint_file [generator checkpoint file path]
   ```
   
   Generated wav files are saved in `generated_files_from_mel` by default.<br>
   You can change the path by adding `--output_dir` option.

## Acknowledgements

We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

## References

[1] Kong J, Kim J, Bae J. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis[J]. Advances in Neural Information Processing Systems, 2020, 33: 17022-17033.

[2] S. Liu, Y. Cao, D. Su and H. Meng, "DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion," *2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*, Cartagena, Colombia, 2021, pp. 741-748, doi: 10.1109/ASRU51503.2021.9688219.

[3] Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003
