import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from model.Acoustic.network import UNet
import sys
sys.path.append("../../")
from config import ATTN_HEADS_WHISPER, ATTN_HEADS_F0, ATTN_HEADS_LOUDNESS, MEL_FREQ_BINS, WHISPER_DIM, WHISPER_CHANNELS, \
    MEL_MAX_LENGTH, MEL_PADDING_LENGTH, WHISPER_SEQ, WHISPER_ALIGN


def transpose(x: torch.Tensor) -> torch.Tensor:
    """
        Swap temporal dimension and feature dimension
    """
    return torch.transpose(x, 1, 2)


class DiffusionConverter(nn.Module):
    """
        A converter network based on UNet for DDPM, which use cross-attention mechanism to project conditioning
        acoustic features to the space of mel-spectrograms, perform Conv1d to align the shape of mel-spectrograms and
        whisper feature maps, and finally concatenate acoustic features channel-wise as the input of UNet to denoise
        the mel-spectrogram.
    """
    def __init__(self,
                 use_cross_attention=True,
                 whisper_attn_heads=ATTN_HEADS_WHISPER,
                 f0_attn_heads=ATTN_HEADS_F0,
                 loudness_attn_heads=ATTN_HEADS_LOUDNESS,
                 whisper_alignment_strategy=WHISPER_ALIGN,
                 ):
        super().__init__()

        assert whisper_alignment_strategy in ('offline', 'online'), \
            "Unsupported Whisper alignment strategy, choose from ('offline', 'online')"
        self.whisper_alignment_strategy = whisper_alignment_strategy

        # backbone denoising network
        self.unet = UNet()

        # cross-attention module
        self.cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention_whisper = nn.MultiheadAttention(embed_dim=WHISPER_DIM, num_heads=whisper_attn_heads,
                                                                 kdim=MEL_FREQ_BINS, vdim=MEL_FREQ_BINS,
                                                                 batch_first=True)
            self.cross_attention_f0 = nn.MultiheadAttention(embed_dim=1, num_heads=f0_attn_heads,
                                                            kdim=MEL_FREQ_BINS, vdim=MEL_FREQ_BINS,
                                                            batch_first=True)
            self.cross_attention_loudness = nn.MultiheadAttention(embed_dim=1, num_heads=loudness_attn_heads,
                                                                  kdim=MEL_FREQ_BINS, vdim=MEL_FREQ_BINS,
                                                                  batch_first=True)

        # whisper CNN module
        """ align the temporal length of Whisper embedding and Mel-spectrograms, and then reduce the dimension of 
            Whisper to the same size of Mel-spectrogram (increase channels at the same time) """
        self.mel_factor = np.lcm(MEL_MAX_LENGTH, WHISPER_SEQ) // MEL_MAX_LENGTH
        self.whisper_factor = np.lcm(MEL_MAX_LENGTH, WHISPER_SEQ) // WHISPER_SEQ
        # temporal alignment: 'online strategy' to align the temporal length
        if whisper_alignment_strategy == 'online':
            self.whisper_conv1d_temporal = nn.Sequential(OrderedDict([
                ('temporal_up_sample',
                 nn.ConvTranspose1d(WHISPER_DIM, WHISPER_DIM, self.whisper_factor, self.whisper_factor)),
                ('temporal_down_sample',
                 nn.Conv1d(WHISPER_DIM, WHISPER_DIM, self.mel_factor, self.mel_factor)),
            ]))
        # dimensionality reduction
        # 1d convolution
        self.whisper_conv1d = nn.Conv1d(WHISPER_DIM, MEL_FREQ_BINS*4, 1, 1)
        # 2d convolution
        self.whisper_conv2d = nn.Sequential(OrderedDict([
            # dimensionality reduction
            ('dim_reduction_1', nn.Conv2d(in_channels=1, out_channels=WHISPER_CHANNELS // 2,
                                          kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))),
            ('dim_reduction_2', nn.Conv2d(in_channels=WHISPER_CHANNELS // 2, out_channels=WHISPER_CHANNELS,
                                          kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))),
        ]))

    def forward(self, xt, t, whisper, f0, loudness):
        """
        Args:
            xt: noisy mel-spectrograms at step t
            t: diffusion step of (batch, 1)
            whisper: whisper embedding of (batch, whisper_dim, whisper_seq)
            f0: f0 of (batch, mel_padding_length)
            loudness: loudness of (batch, mel_padding_length)

        Returns:
            denoised mel-spectrograms
        """
        f0 = f0.unsqueeze(1)
        loudness = loudness.unsqueeze(1)
        if self.whisper_alignment_strategy == 'offline':
            whisper = whisper.repeat_interleave(self.whisper_factor, 2)
            whisper = F.avg_pool1d(whisper, self.mel_factor, self.mel_factor)
        else:
            whisper = self.whisper_conv1d_temporal(whisper)

        if self.cross_attention:
            whisper, _ = self.cross_attention_whisper(transpose(whisper), transpose(xt), transpose(xt))
            whisper = transpose(whisper)
            f0, _ = self.cross_attention_f0(transpose(f0), transpose(xt), transpose(xt))
            f0 = transpose(f0)
            loudness, _ = self.cross_attention_loudness(transpose(loudness), transpose(xt), transpose(xt))
            loudness = transpose(loudness)

        whisper = self.whisper_conv1d(whisper)
        whisper = self.whisper_conv2d(whisper.unsqueeze(1))
        xt = xt.unsqueeze(1)
        f0 = f0.unsqueeze(1).repeat(1, 1, MEL_FREQ_BINS, 1)
        loudness = loudness.unsqueeze(1).repeat(1, 1, MEL_FREQ_BINS, 1)

        return torch.squeeze(self.unet(torch.concat((xt, whisper, f0, loudness), dim=1), t), 1)


# for testing functionality
if __name__ == '__main__':
    # cross-attention module
    def transpose(x):
        return torch.swapaxes(x, 1, 2)

    query_dim = 512
    key_dim = 1
    key = torch.randn((8, key_dim, 496))
    value = key
    query = torch.randn((8, query_dim, 400))
    cross_attention = nn.MultiheadAttention(batch_first=True, embed_dim=query_dim, num_heads=4,
                                            kdim=key_dim, vdim=key_dim)
    test, _ = cross_attention(transpose(query), transpose(key), transpose(value))
    print(transpose(test).shape)

    # Whisper convolution module
    whisper_conv = nn.Sequential(OrderedDict([
        # 'Online strategy' to align the temporal length
        # ('temporal_up_sample', nn.ConvTranspose1d(WHISPER_DIM, WHISPER_DIM, 5, 5)),
        # ('temporal_down_sample', nn.Conv1d(WHISPER_DIM, WHISPER_DIM, 4, 4)),
        # dimensionality reduction
        ('dim_reduction_1', nn.Conv1d(in_channels=WHISPER_DIM, out_channels=MEL_FREQ_BINS*4,
                                      kernel_size=3, stride=1, padding=1)),
        ('dim_reduction_2', nn.Conv2d(in_channels=1, out_channels=WHISPER_CHANNELS // 2,
                                      kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))),
        ('dim_reduction_3', nn.Conv2d(in_channels=WHISPER_CHANNELS // 2, out_channels=WHISPER_CHANNELS,
                                      kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))),
    ]))
    test = whisper_conv(torch.randn((1, 512, 400)))
    print(test.size())

    # off-line alignment strategy
    test = np.arange(512*400)
    test = torch.Tensor(test.reshape(512, 400))
    test = test.repeat_interleave(5, 1)
    print(test)
    test_result_1 = torch.mean(test.T.reshape(-1, 4, 512), dim=1).T.numpy()
    test_result_2 = F.avg_pool1d(test, 4, 4).numpy()
    print(np.sum(test_result_1 == test_result_2 - 1))
