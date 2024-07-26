import math
import torch
from .functions import normal_kl, discretized_gaussian_loglik, flat_mean

from .diffusion import GaussianDiffusion

# The following code is adapted from DDNM
###############################################
# def MeanUpsample(x, scale):
#     n, c, h, w = x.shape
#     out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
#     out = out.view(n, c, scale*h, scale*w)
#     return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)

def get_degradation_operator(deg_type, chan=3, res=32, device=torch.device("cpu")):
    # get degradation operator
    print("deg_type:",deg_type)
    if deg_type =='colorization':
        A = lambda z: color2gray(z)
        Ap = lambda z: gray2color(z)
    elif deg_type =='denoising':
        A = lambda z: z
        Ap = A
    elif deg_type =='sr_averagepooling':
        raise NotImplementedError("sr not useful for CIFAR10")
        # scale=round(deg_type_scale)
        # A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        # Ap = lambda z: MeanUpsample(z,scale)
    elif deg_type =='inpainting':
        # blocks 1/3 to 1/2 of pixels in height and width
        mask = torch.ones(chan, res, res, device=device, dtype=torch.float32)
        mask[:, res//4:2*res//3, res//3:res//2] = 0
        Ap = A = lambda z: z*mask
    # elif deg_type =='all':
    #     loaded = np.load("inp_masks/mask.npy")
    #     mask = torch.from_numpy(loaded).to(self.device)
    #     A1 = lambda z: z*mask
    #     A1p = A1
    #
    #     A2 = lambda z: color2gray(z)
    #     A2p = lambda z: gray2color(z)
    #
    #     scale=deg_type_scale
    #     A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    #     A3p = lambda z: MeanUpsample(z,scale)
    #
    #     A = lambda z: A3(A2(A1(z)))
    #     Ap = lambda z: A1p(A2p(A3p(z)))
    else:
        raise NotImplementedError("degradation type not supported")
    return A, Ap
###############################################

class ConditionalGaussianDiffusion(GaussianDiffusion):
    def __init__(self, betas, H, Hp, sigma_y, deg_type, **diffusion_kwargs):
        model_mean_type = diffusion_kwargs["model_mean_type"]
        model_var_type = diffusion_kwargs["model_var_type"]
        loss_type = diffusion_kwargs["loss_type"]
        assert(model_var_type in ["fixed-small", "fixed-large"])
        assert(model_mean_type == "eps")
        super().__init__(betas, **diffusion_kwargs)
        self.deg_type = deg_type
        self.alphas = 1 - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.step_posterior_mean_coef1 = betas * self.sqrt_recip_alphas
        self.step_posterior_mean_coef2 = betas * self.sqrt_recip_alphas / self.sqrt_one_minus_alphas_bar
        self.one_minus_alphas_bar_plus_noise = 1. - self.alphas_bar + self.alphas_bar * sigma_y**2)
        self.H = H
        self.Hp = Hp

    def p_cond_mean_var(self, denoise_fn, x_t, t, y, clip_denoised, return_pred):
        B, C, H, W = x_t.shape
        out = denoise_fn(x_t, t)

        if self.model_var_type == "learned":
            assert all(out.shape == (B, 2 * C, H, W))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t),\
                                      self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the conditional mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        if self.model_mean_type == "mean":
            pred_x_0 = _clip(self._pred_x_0_from_mean(x_t=x_t, mean=out, t=t))
            pred_x_0 = self.Hp(y) + pred_x_0 - self.Hp(self.H(pred_x_0))
            model_mean = out
        elif self.model_mean_type == "x_0":
            pred_x_0 = _clip(out)
            pred_x_0 = self.Hp(y) + pred_x_0 - self.Hp(self.H(pred_x_0))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        elif self.model_mean_type == "eps":
            pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            pred_x_0 = self.Hp(y) + pred_x_0 - self.Hp(self.H(pred_x_0))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    def p_cond_sample_step(self, denoise_fn, x_t, t, y, clip_denoised=True, return_pred=False, generator=None):
        clip_denoised = False
        model_mean, _, model_logvar, pred_x_0 = self.p_cond_mean_var(
            denoise_fn, x_t, t, y, clip_denoised=clip_denoised, return_pred=True)
        model_mean_test, *_ = self.p_cond_mean_var_noisy(
            denoise_fn, x_t, t, y, clip_denoised=clip_denoised)
        print(model_mean)
        print(model_mean_test)
        noise = torch.empty_like(x_t).normal_(generator=generator)
        nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise
        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_cond_sample(self, denoise_fn, y, div=1, shape=None, device=torch.device("cpu"), noise=None, seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        y = y.to(device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        for ti in range(self.timesteps - div, -1, -div):
            t.fill_(ti)
            x_t = self.p_cond_sample_step(denoise_fn, x_t, t, y, generator=rng)
        return x_t

    def p_cond_mean_var_noisy(self, denoise_fn, x_t, t, y, clip_denoised):
        if clip_denoised:
            raise NotImplementedError(clip_denoised)

        out = denoise_fn(x_t, t)

        if self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t),\
                                      self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.deg_type not in ["inpainting", "denoising"]:
            raise NotImplementedError(self.deg_type)

        # calculate the conditional mean estimate
        if self.model_mean_type == "eps":
            # pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            # pred_x_0 = self.Hp(y) + pred_x_0 - self.Hp(self.H(pred_x_0))
            # model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
            coef1 = self._extract(self.sqrt_recip_alphas, t, x_t)
            coef2 = self._extract(self.step_posterior_mean_coef2, t, x_t)
            coef3 = self._extract(self.step_posterior_mean_coef1, t, x_t)
            f_y = (self.sqrt_alphas_bar * self.Hp(y) - self.H(self.Hp(x_t))) / self.one_minus_alphas_bar_plus_noise
            model_mean = coef1 * x_t - coef2 * (out - self.Hp(self.H(out))) + coef3 * f_y
        else:
            raise NotImplementedError(self.model_mean_type)

        return model_mean, model_var, model_logvar

    def p_cond_sample_noisy_step(self, denoise_fn, x_t, t, y, clip_denoised=False, generator=None):
        model_mean, _, model_logvar = self.p_cond_mean_var_noisy(
            denoise_fn, x_t, t, y, clip_denoised=clip_denoised)
        noise = torch.empty_like(x_t).normal_(generator=generator)
        nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise
        return sample

    @torch.inference_mode()
    def p_cond_sample_noisy(self, denoise_fn, y, div=1, shape=None, device=torch.device("cpu"), noise=None, seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        y = y.to(device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        for ti in range(self.timesteps - div, -1, -div):
            t.fill_(ti)
            x_t = self.p_cond_sample_noisy_step(denoise_fn, x_t, t, y, generator=rng)
        return x_t
