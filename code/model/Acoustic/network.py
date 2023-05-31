import warnings
from inspect import isfunction
from functools import partial
import math
import torch
from torch import nn, einsum
from einops import rearrange
import sys
sys.path.append("../../")
from config import CHANNELS_BASE, CHANNELS_MULT_FACTORS, CHANNELS_INPUT, CHANNELS_OUTPUT, \
    BASIC_BLOCK, POSITION_ENC_DIM, SHAPE_CHANGE


# ################################# Denoising Network: U-Net ##################################
# Primary Reference (borrowed, modified and annotated):
# http://www.egbenz.com/#/my_article/19
#
# Other possible references:
# https://github.com/bubbliiiing/ddpm-pytorch
# https://github.com/lucidrains/denoising-diffusion-pytorch

def get_norm(norm, num_channels=None, num_groups=None, layer_shape=None):
    """
        get normalization layers from pytorch module: ('instance', 'batch', 'group', 'layer', 'simple_layer', None)
    """
    if norm == "instance":
        assert exists(num_channels), 'num_channels of the InstanceNorm is not specified.'
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "batch":
        assert exists(num_channels), 'num_channels of the BatchNorm is not specified.'
        return nn.BatchNorm2d(num_channels)
    elif norm == "group":
        assert exists(num_channels), 'num_channels of the GroupNorm is not specified.'
        assert exists(num_groups), 'num_groups of the GroupNorm is not specified.'
        return nn.GroupNorm(num_groups, num_channels)
    elif norm == 'layer':
        assert exists(layer_shape), 'normalized_shape of the LayerNorm is not specified.'
        return nn.LayerNorm(layer_shape)
    # simple layer normalization layer with bias and scale shared across all location
    elif norm == 'simple_layer':
        assert exists(num_channels), 'num_channels of the Simple LayerNorm is not specified.'
        return nn.GroupNorm(1, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


def exists(x):
    """ check the existence of an input variable """
    return x is not None


def default(val, d):
    """
        add default value to a variable

    Args:
        val: target variable
        d: default value

    Returns:
        'val' with a default value of 'd'
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    """
        Residual Connection Module
    """
    def __init__(self, fn):
        """
            Residual Connection Module
        Args:
            fn: nn.Module to be wrapped by residual connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
            Residual Connected output of inner module
        Args:
            x: input

        Returns:
            Residual Connected output of inner module
        """
        return self.fn(x, *args, **kwargs) + x


class Swish(nn.Module):
    """
        Swish activation function: x * Sigmoid(x)
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    """
        Sinusoidal Positional Encodings of diffusion steps;
        INPUT: diffusion steps tensor of shape (batch, 1);
        OUTPUT: diffusion steps embedding tensor of shape (batch, dim).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
            Sinusoidal Positional Encodings of diffusion steps

        Args:
            time: diffusion steps tensor of shape (batch, 1)

        Returns:
            diffusion steps embedding tensor of shape (batch, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def UpSample(dim, shape_change=True):
    """
        Up-sampling block for UNet
        Different traditional UNet, change of channel dimension is done during extraction block

    Args:
        dim: output dimension of up-sampling block
        shape_change: whether to shrink the height and width of the input feature map by 0.5 or not

    Returns:
        an up-sampling block nn.Module
    """
    if shape_change:
        return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    else:
        return nn.ConvTranspose2d(dim, dim, 5, 1, 2)


def DownSample(dim, shape_change=True):
    """
        Down-sampling block for UNet
        Different traditional UNet, change of channel dimension is done during extraction block

    Args:
        dim: output dimension of down-sampling block
        shape_change: whether to spand the height and width of the input feature map by 2 or not

    Returns:
        a down sampling block nn.Module
    """
    if shape_change:
        return nn.Conv2d(dim, dim, 4, 2, 1)
    else:
        return nn.Conv2d(dim, dim, 5, 1, 2)


class ConvBlock(nn.Module):
    """
        Basic conv sub-block for ResNet:
        conv2d -> GroupNorm -> activation
    """

    def __init__(self, dim, dim_out, groups=8, activation=nn.SiLU()):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = activation

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
        Basic block of ResNet tailored for DDPM UNet.
        https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, activation=nn.SiLU()):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = ConvBlock(dim, dim_out, groups=groups, activation=activation)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups, activation=activation)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNeXtBlock(nn.Module):
    """
        Basic block of ConvNeXt tailored for DDPM UNet.
        A ConvNet for the 2020s, https://arxiv.org/abs/2201.03545
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True,
                 activation_extraction=nn.GELU(), activation_time=Swish()):
        """
            Initializing a ConvNeXt block for DDPM UNet.

        Args:
            dim: input dimension of channels
            dim_out: output dimension of channels
            time_emb_dim: dimension of the time embedding
            mult: multiplying factor for the Inverted Bottleneck channel dimension
            norm: whether to use normalization layer
            activation_extraction: activation function for the feature extraction layers
            activation_time: activation function for the MLP to process time embedding
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(activation_time, nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        # Depth-wise Convolution layer with large kernel
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        # Inverted Bottleneck module
        self.net = nn.Sequential(
            get_norm('simple_layer', dim) if norm else get_norm(None),      # extra normalization layer
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            activation_extraction,
            get_norm('simple_layer', dim_out * mult),           # the sole normalization layer used in origin paper
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        # 1x1 conv for residual connection with different channel dimension
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
        MSA Layer for processing bottleneck
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
        Linear MSA Layer for down-sampling and up-sampling
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """
        Group Normalization for nn.Module wrapper
    """
    def __init__(self, dim, fn, num_groups=1):
        """
            Group Normalization nn.Module wrapper

        Args:
            dim: input channel dimension
            fn: inner nn.Module
            num_groups: number of groups
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class UNet(nn.Module):
    """
        UNet tailored for the DDPM
    """
    def __init__(
            self,
            in_channels=CHANNELS_INPUT,
            out_channels=CHANNELS_OUTPUT,
            base_channels=CHANNELS_BASE,
            channels_mults=CHANNELS_MULT_FACTORS,
            init_channels=CHANNELS_BASE,
            with_time_emb=True,
            position_enc_dim=POSITION_ENC_DIM,
            time_emd_dim=None,
            basic_block=BASIC_BLOCK,
            resnet_block_groups=8,
            convnext_mult=2,
            activation_time=Swish(),
            activation_klass=nn.GELU(),
            msa_heads=4,
            msa_head_dim=32,
    ):
        """
            Initialize UNet as denoising network for DDPM.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            base_channels: base output channel dimension for sequentially multiplying
            channels_mults: output channel dimension multiplying factors during down-sampling
            init_channels: output channel dimension for the initial convolution block
            with_time_emb: whether to use time embedding
            position_enc_dim: dimension of raw positional embedding of the diffusion steps
            time_emd_dim: dimension of time embedding, processed from positional embedding with an MLP
            basic_block: type of basic blocks for feature extraction, ('resnet', 'convnext')
            resnet_block_groups: number of ResNeXt channel groups
            convnext_mult: multiplying factor for the channel dimension of inverted bottleneck module for ConvNeXt
            activation_time: activation function for the time embedding processing MLP
            activation_klass: activation function for the feature extraction blocks
            msa_heads: number of heads for the MSA Layer
            msa_head_dim: dimension of each head in the MSA Layer
        """
        super().__init__()

        # determine input channel dimensions
        self.channels = in_channels

        # initial convolution
        init_channels = default(init_channels, base_channels // 3 * 2)
        self.init_conv = nn.Conv2d(in_channels, init_channels, 7, padding=3)

        # calculate output channel dimensions for each block
        dims = [init_channels, *map(lambda m: base_channels * m, channels_mults)]
        # (input channel dimension, output channel dimension) for each block
        in_out = list(zip(dims[:-1], dims[1:]))

        # basic feature extraction block connecting down-sampling blocks, middle blocks and up-sampling blocks
        if basic_block == 'convnext':
            block_klass = partial(ConvNeXtBlock, mult=convnext_mult,
                                  activation_extraction=activation_klass, activation_time=activation_time)
        elif basic_block == 'resnet':
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)
            warnings.warn('ResNet backbone for the UNet is not maintained', DeprecationWarning)
        else:
            raise ValueError("Unsupported basic block type for UNet! ('resnet', 'convnext')")

        # time embeddings block, processing raw Positional Embedding with an MLP to get time embedding
        time_emd_dim = default(time_emd_dim, 4 * base_channels)
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                PositionalEncoding(position_enc_dim),
                nn.Linear(position_enc_dim, time_emd_dim),
                activation_time,
                nn.Linear(time_emd_dim, time_emd_dim),
            )
        else:
            time_emd_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # down-sampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_emd_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_emd_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=msa_heads, dim_head=msa_head_dim))),
                        DownSample(dim_out, SHAPE_CHANGE) if not is_last else nn.Identity(),
                    ]
                )
            )

        # middle blocks for higher-level feature extraction based on the bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_emd_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=msa_heads, dim_head=msa_head_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_emd_dim)

        # up-sampling blocks (abandon the last one when up-sampling)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_emd_dim),      # '*2' is due to concatenation
                        block_klass(dim_in, dim_in, time_emb_dim=time_emd_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=msa_heads, dim_head=msa_head_dim))),
                        UpSample(dim_in, SHAPE_CHANGE) if not is_last else nn.Identity(),
                    ]
                )
            )

        # final convolution block
        out_channels = default(out_channels, in_channels)
        self.final_conv = nn.Sequential(
            block_klass(init_channels, init_channels), nn.Conv2d(init_channels, out_channels, 1)
        )

    def forward(self, x, time):
        """
            Parameterize target variable with denoising UNet during diffusion process.
        Args:
            x: noisy input a diffusion step t
            time: the diffusion step t

        Returns:
            Approximated target variable (e.g., epsilon)
        """
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # down-sample
        for block1, block2, attn, down_sample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = down_sample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # up-sample
        count = 0
        for block1, block2, attn, up_sample in self.ups:
            count += 1
            print('Up-sampling block', count)
            print('Shape of x', x.shape)
            print('Shapes of h')
            for temp in h:
                print(temp.shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = up_sample(x)

        print(x.shape)
        return self.final_conv(x)


# for testing functionality
if __name__ == '__main__':
    channel_dimensions = [CHANNELS_BASE, *map(lambda m: CHANNELS_BASE * m, CHANNELS_MULT_FACTORS)]
    print(channel_dimensions)
    in_out = list(zip(channel_dimensions[:-1], channel_dimensions[1:]))
    print(in_out)
    print(list(reversed(in_out)))
