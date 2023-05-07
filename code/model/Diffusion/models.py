from copy import deepcopy
from collections import OrderedDict
from inspect import isfunction
from functools import partial
import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import sys

sys.path.append("../")
from config import CHANNELS_BASE, CHANNELS_MULT_FACTORS, DIFFUSION_STEPS, LINEAR_BETA_1, LINEAR_BETA_T, BASIC_BLOCK, \
    POSITION_EMBED_DIM, CHANNELS_INPUT, CHANNELS_OUTPUT


# ################################# Denoising Diffusion Probabilistic Model #################################
# Borrowed, modified and annotated from: https://github.com/w86763777/pytorch-ddpm

def get_cosine_noise_schedule(steps, s=0.008):
    """
        Cosine Noise Scheduling Rate: https://arxiv.org/abs/2102.09672
    """
    f_t = np.power(np.cos(np.arange(steps + 1) / steps + s) / (1 + s) * np.pi / 2, 2)
    return torch.clip(torch.Tensor(f_t / f_t[0]), 0.0001, 0.9999)


def extract(v, t, x_shape):
    """
        Extract some coefficients at specified time steps,
                [by 'torch.gather()']
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
                [by Tensor.view(), analogous to ndarray.reshape()]
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    """ Training Denoising Network with Diffusion Forward Process via ELBO """

    def __init__(self, model, T, noise_schedule='cosine', beta_1=LINEAR_BETA_1, beta_T=LINEAR_BETA_T):
        """
        Initializing a DDPM forward training process.
        Args:
            model: denoising network
            T: number of diffusion steps
            noise_schedule: types of noise scheduling scheme, ('linear', 'cosine')
            beta_1: start of linear noise scheduling scheme
            beta_T: end of linear noise scheduling scheme
        """
        super().__init__()

        self.model = model
        self.T = T

        # register parameters for noise scheduling, for diffusion q(x_t | x_{t-1}) and others
        if noise_schedule == 'cosine':
            alphas_bar = get_cosine_noise_schedule(T)
        elif noise_schedule == 'linear':
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
        else:
            raise ValueError("Unsupported noise scheduling scheme! Choose from ('linear', 'cosine').")
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
            Algorithm 1. Training
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    """ Reverse Diffusion Process with Gaussian Noise """

    def __init__(self, model, T, noise_schedule='cosine', beta_1=LINEAR_BETA_1, beta_T=LINEAR_BETA_T,
                 mean_type='epsilon', var_type='fixedlarge'):
        """
        Initializing a DDPM inference process, based on Langevin Dynamics for reverse diffusion
        Args:
            model: (trained) denoising network
            T: number of diffusion steps
            noise_schedule: types of noise scheduling scheme, ('linear', 'cosine')
            beta_1: start of linear noise scheduling scheme
            beta_T: end of linear noise scheduling scheme
            mean_type: elements for the denoising network to parameterize, ('xprev' 'xstart', 'epsilon')
            var_type: types of variance
        """
        assert noise_schedule in ['linear', 'cosine'], \
            "Unsupported noise scheduling scheme! Choose from ('linear', 'cosine')."
        assert mean_type in ['xprev' 'xstart', 'epsilon'], \
            "Invalid mean_type input!"
        assert var_type in ['fixedlarge', 'fixedsmall'], \
            "Invalid var_type input!"
        super().__init__()

        self.model = model
        self.T = T
        self.mean_type = mean_type
        self.var_type = var_type

        # register parameters for noise scheduling
        if noise_schedule == 'cosine':
            alphas_bar = get_cosine_noise_schedule(T)
            alphas = torch.zeros(len(alphas_bar))
            for i in reversed(range(1, len(alphas))):
                alphas[i] = alphas_bar[i] / alphas_bar[i - 1]
            alphas[0] = alphas_bar[0]
            self.register_buffer('betas', 1 - alphas)
        elif noise_schedule == 'linear':
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
        else:
            raise ValueError("Unsupported noise scheduling scheme! Choose from ('linear', 'cosine').")
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # alphas_bar from last time step

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0): Langevin Dynamics
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_var_clipped',
                             torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
            Langevin Dynamics.
            Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_x0_from_eps(self, x_t, t, eps):
        """
            Calculation of x_0 using network approximated epsilon (noise).
            For 'epsilon'-type network parameterization.
        """
        assert x_t.shape == eps.shape
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_x0_from_xprev(self, x_t, t, xprev):
        """
            Calculation of x_0 using network approximated x_(t-1). [or x_(t+1)? don't know.]
            For 'xprev'-type network parameterization.
        """
        assert x_t.shape == xprev.shape
        # Formula: (xprev - coef2*x_t) / coef1
        return (
                extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        """
            Network parameterized / approximated Langevin Dynamics.

        Args:
            x_t: noisy input at step t.
            t: diffusion step.

        Returns:
            Approximated (mean, variance) for diffusion reverse process.
        """
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':  # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_x0_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':  # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':  # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_x0_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
            Algorithm 2. Sampling
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t

        # Clipped to [-1, 1] in the  original DDPM paper.
        # The authors assume that the image input is already rescaled to [-1, 1] to better fit standard Gaussian noise.
        return torch.clip(x_0, -1, 1)


class EMA(nn.Module):
    """
        Wrapper for building Pytorch model with Exponential Moving Average.
        Borrowed and modified from: https://www.zijianhu.com/post/pytorch/ema/
    """

    def __init__(self, model: nn.Module, decay=0.999):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=sys.stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, **inputs):
        if self.training:
            return self.model(**inputs)
        else:
            return self.shadow(**inputs)


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
        assert exists(layer_shape), 'num_channels of the Simple LayerNorm is not specified.'
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


class PositionalEmbedding(nn.Module):
    """
        Sinusoidal Positional Embeddings of diffusion steps;
        INPUT: diffusion steps tensor of shape (batch, 1);
        OUTPUT: diffusion steps embedding tensor of shape (batch, dim).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
            Sinusoidal Positional Embeddings of diffusion steps

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


def UpSample(dim):
    """
        Up-sampling block for UNet
        Different traditional UNet, change of channel dimension is done during extraction block

    Args:
        dim: output dimension of up-sampling block

    Returns:
        an up-sampling block nn.Module
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def DownSample(dim):
    """
        Down-sampling block for UNet
        Different traditional UNet, change of channel dimension is done during extraction block

    Args:
        dim: output dimension of down-sampling block

    Returns:
        a down sampling block nn.Module
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)


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
            position_emb_dim=POSITION_EMBED_DIM,
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
            position_emb_dim: dimension of raw positional embedding of the diffusion steps
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
            block_klass = partial(ConvNeXtBlock, mult=convnext_mult, activation=activation_klass)
        elif basic_block == 'resnet':
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        else:
            raise ValueError("Unsupported basic block type for UNet! ('resnet', 'convnext')")

        # time embeddings block, processing raw Positional Embedding with an MLP to get time embedding
        time_emd_dim = default(time_emd_dim, 4 * base_channels)
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                PositionalEmbedding(position_emb_dim),
                nn.Linear(position_emb_dim, time_emd_dim),
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
                        DownSample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # middle blocks for higher-level feature extraction based on the bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_emd_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=msa_heads, dim_head=msa_head_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_emd_dim)

        # up-sampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_emd_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_emd_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=msa_heads, dim_head=msa_head_dim))),
                        UpSample(dim_in) if not is_last else nn.Identity(),
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
        for block1, block2, attn, up_sample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = up_sample(x)

        return self.final_conv(x)


# for testing functionality
if __name__ == '__main__':
    alpha_bar = get_cosine_noise_schedule(DIFFUSION_STEPS)
    alpha = torch.zeros(len(alpha_bar))
    for i in reversed(range(1, len(alpha))):
        alpha[i] = alpha_bar[i] / alpha_bar[i - 1]
    alpha[0] = alpha_bar[0]
    print('alpha_bar', alpha_bar)
    print('alpha', alpha)

    channel_dimensions = [CHANNELS_BASE, *map(lambda m: CHANNELS_BASE * m, CHANNELS_MULT_FACTORS)]
    print(channel_dimensions)
    print(list(zip(channel_dimensions[:-1], channel_dimensions[1:])))
