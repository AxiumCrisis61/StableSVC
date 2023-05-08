import torch
from torch import nn
from network import UNet
import sys
sys.path.append("../../")
from config import ATTN_HEADS_WHISPER, ATTN_HEADS_F0, ATTN_HEADS_LOUDNESS, MEL_FREQ_BINS, WHISPER_DIM


def transpose(x: torch.Tensor) -> torch.Tensor:
    """
        Swap temporal dimension and feature dimension
    """
    return torch.transpose(x, 1, 2)


class DiffusionConverter(nn.Module):
    """
        A converter network based on UNet for DDPM, which use cross-attention mechanism to project conditioning
        acoustic features to the space of mel-spectrograms, perform Conv1d to align the shape of mel-spectrograms and
        whisper feature maps, and finally concatenate acoustics features channel-wise as the input of UNet to denoise
        the mel-spectrogram.
    """
    def __init__(self,
                 cross_attention=True,
                 whisper_attn_heads=ATTN_HEADS_WHISPER,
                 f0_attn_heads=ATTN_HEADS_F0,
                 loudness_attn_heads=ATTN_HEADS_LOUDNESS,
                 ):
        super().__init__()
        self.unet = UNet()
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attention_whisper = nn.MultiheadAttention(WHISPER_DIM, whisper_attn_heads, batch_first=True)
            self.cross_attention_f0 = nn.MultiheadAttention(1, f0_attn_heads, batch_first=True)
            self.cross_attention_loudness = nn.MultiheadAttention(1, loudness_attn_heads, batch_first=True)
        self.whisper_conv = nn.Conv1d(WHISPER_DIM, MEL_FREQ_BINS, 3, 1, 1)

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

        if self.cross_attention:
            whisper, _ = transpose(self.cross_attention_whisper(transpose(whisper), transpose(xt), transpose(xt)))
            f0, _ = transpose(self.cross_attention_f0(transpose(f0), transpose(xt), transpose(xt)))
            loudness, _ = transpose(self.cross_attention_loudness(transpose(loudness), transpose(xt), transpose(xt)))
        whisper = self.whisper_conv(whisper)

        xt = xt.unsqueeze(1)
        whisper = whisper.unsqueeze(1)
        f0 = f0.unsqueeze(1).repeat(2, MEL_FREQ_BINS)
        loudness = loudness.unsqueeze(1).repeat(2, MEL_FREQ_BINS)

        return UNet(torch.concat((xt, whisper, f0, loudness), dim=1), t)
