import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("../../")
from config import DIFFUSION_STEPS, LINEAR_BETA_1, LINEAR_BETA_T


# ################################# Basic Denoising Diffusion Probabilistic Model #################################
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
    """ Training Denoising Network with Acoustic Forward Process via ELBO """

    def __init__(self, model, T=DIFFUSION_STEPS, noise_schedule='cosine', beta_1=LINEAR_BETA_1, beta_T=LINEAR_BETA_T):
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

    def forward(self, x_0, **features):
        """
            Algorithm 1. Training

        Args:
            x_0: original input
            **features: other features or parameters for the network (passed as dictionary)

        Returns:
            loss value
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t, **features), noise, reduction='none').mean()
        return loss


class GaussianDiffusionSampler(nn.Module):
    """ Reverse Acoustic Process with Gaussian Noise """

    def __init__(self, model, T=DIFFUSION_STEPS, noise_schedule='cosine', beta_1=LINEAR_BETA_1, beta_T=LINEAR_BETA_T,
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
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T+1]  # alphas_bar from last time step

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

    def p_mean_variance(self, x_t, t, **features):
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
            x_prev = self.model(x_t, t, **features)
            x_0 = self.predict_x0_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':  # the model predicts x_0
            x_0 = self.model(x_t, t, **features)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':  # the model predicts epsilon
            eps = self.model(x_t, t, **features)
            x_0 = self.predict_x0_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, **features):
        """
            Algorithm 2. Sampling

        Args:
            x_T: Gaussian Noise
            **features: other features or parameters for the network (passed as dictionary)

        Returns:
            sampled diffusion generation
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, **features)
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


# for testing functionality
if __name__ == '__main__':
    alpha_bar = get_cosine_noise_schedule(DIFFUSION_STEPS)
    alpha = torch.zeros(len(alpha_bar))
    for i in reversed(range(1, len(alpha))):
        alpha[i] = alpha_bar[i] / alpha_bar[i - 1]
    alpha[0] = alpha_bar[0]
    print('alpha_bar', alpha_bar)
    print('alpha', alpha)

    alpha_bar_prev = F.pad(alpha_bar, [1, 0], value=1)[:DIFFUSION_STEPS+1]
    print('alpha_bar_prev', alpha_bar_prev)
    print(alpha.shape == alpha_bar_prev.shape)
