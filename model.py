'''
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''
import math

import torch
import torch.nn as nn

import bayesian_layers as bl
from model import TensorDataset

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False, mle: bool = False, logvar_init: float = -8.0
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = bl.Conv2d(in_channels, out_channels, 3, 1, 1, mle=mle, logvar_init=logvar_init)
        self.after_conv1 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = bl.Conv2d(out_channels, out_channels, 3, 1, 1, mle=mle, logvar_init=logvar_init)
        self.after_conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, num_param_samples=10) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x, num_param_samples)
            x1 = self.after_conv1(x1)
            x2 = self.conv2(x1, num_param_samples)
            x2 = self.after_conv2(x2)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x, num_param_samples)
            x1 = self.after_conv1(x1)
            x2 = self.conv2(x1, num_param_samples)
            x2 = self.after_conv2(x2)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, mle=False, logvar_init=-8.0):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        self.res = ResidualConvBlock(in_channels, out_channels, mle=mle, logvar_init=logvar_init)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, num_param_samples=10):
        x = self.res(x, num_param_samples)
        return self.maxpool(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, mle=False, logvar_init=-8.0):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        self.ct = bl.ConvTranspose2d(in_channels, out_channels, 2, 2, mle=mle, logvar_init=logvar_init)
        self.r1 = ResidualConvBlock(out_channels, out_channels, mle=mle, logvar_init=logvar_init)
        self.r2 = ResidualConvBlock(out_channels, out_channels, mle=mle, logvar_init=logvar_init)

    def forward(self, x, skip, num_param_samples=10):
        x = torch.cat((x, skip), 1)
        x = self.ct(x, num_param_samples)
        x = self.r1(x, num_param_samples)
        return self.r2(x, num_param_samples)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, mle=False, logvar_init=-8.0):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        self.l1 = bl.Linear(input_dim, emb_dim, mle=mle, logvar_init=logvar_init)
        self.l2 = bl.Linear(emb_dim, emb_dim, mle=mle, logvar_init=logvar_init)

    def forward(self, x, num_param_samples=10):
        x = x.view(-1, self.input_dim)
        x = nn.functional.gelu(self.l1(x, num_param_samples))
        return self.l2(x, num_param_samples)


class EmbedFC_deterministic(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC_deterministic, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        self.l1 = nn.Linear(input_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, num_param_samples=10):
        x = x.view(-1, self.input_dim)
        x = nn.functional.gelu(self.l1(x))
        return self.l2(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10, mle=False, deterministic_embed=False, logvar_init=-8.0):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.mle = mle
        self.deterministic_embed = deterministic_embed

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True, mle=mle, logvar_init=logvar_init)

        self.down1 = UnetDown(n_feat, n_feat, mle=mle, logvar_init=logvar_init)
        self.down2 = UnetDown(n_feat, 2 * n_feat, mle=mle, logvar_init=logvar_init)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat, mle=mle, logvar_init=logvar_init)
        self.timeembed2 = EmbedFC(1, 1*n_feat, mle=mle, logvar_init=logvar_init)

        if self.deterministic_embed:
            self.contextembed1 = nn.ModuleList([EmbedFC_deterministic(1, 2*n_feat) for _ in range(n_classes)])
            self.contextembed2 = nn.ModuleList([EmbedFC_deterministic(1, 1*n_feat) for _ in range(n_classes)])
            self.c1_det_mean = nn.Parameter(torch.Tensor(1, 2*n_feat))
            self.c2_det_mean = nn.Parameter(torch.Tensor(1, 1*n_feat))
            self.c1_det_logvar = nn.Parameter(torch.Tensor(1, 2*n_feat))
            self.c2_det_logvar = nn.Parameter(torch.Tensor(1, 1*n_feat))
            nn.init.kaiming_uniform_(self.c1_det_mean, a=math.sqrt(5))
            nn.init.constant_(self.c1_det_logvar, logvar_init)
            nn.init.kaiming_uniform_(self.c2_det_mean, a=math.sqrt(5))
            nn.init.constant_(self.c2_det_logvar, logvar_init)
        else:
            self.contextembed1 = EmbedFC(n_classes, 2*n_feat, mle=mle, logvar_init=logvar_init)
            self.contextembed2 = EmbedFC(n_classes, 1*n_feat, mle=mle, logvar_init=logvar_init)

        # self.timeembed1 = EmbedFC_deterministic(1, 2*n_feat)
        # self.timeembed2 = EmbedFC_deterministic(1, 1*n_feat)
        # self.contextembed1 = EmbedFC_deterministic(n_classes, 2*n_feat)
        # self.contextembed2 = EmbedFC_deterministic(n_classes, 1*n_feat)

        self.up0_ct = bl.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7, mle=mle, logvar_init=logvar_init)
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat, mle=mle, logvar_init=logvar_init)
        self.up2 = UnetUp(2 * n_feat, n_feat, mle=mle, logvar_init=logvar_init)
        self.out_conv1 = bl.Conv2d(2 * n_feat, n_feat, 3, 1, 1, mle=mle, logvar_init=logvar_init)
        self.out = nn.Sequential(
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
        )
        self.out_conv2 = bl.Conv2d(n_feat, self.in_channels, 3, 1, 1, mle=mle, logvar_init=logvar_init)

    def forward(self, x, c, t, context_mask, num_param_samples=10):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = x.repeat(num_param_samples, 1, 1, 1)
        t = t.repeat(num_param_samples, 1, 1, 1)

        x = self.init_conv(x, num_param_samples)
        down1 = self.down1(x, num_param_samples)
        down2 = self.down2(down1, num_param_samples)
        hiddenvec = self.to_vec(down2)

        if self.deterministic_embed:
            one_input = torch.ones((1,1)).type(torch.float).to(c.device)
            c = c + (context_mask) * 10
            class_indices = [torch.where(c==i)[0] for i in range(self.n_classes)]
            mask_indices = torch.where(context_mask==1)[0]

            def context_embed(embed_fns, det_mean, det_logvar, out_shape):
                cemb = torch.empty((c.shape[0], out_shape), dtype=torch.float).to(c.device)
                for i in range(self.n_classes):
                    cemb[class_indices[i]] = embed_fns[i](one_input)
                cemb = cemb.repeat(num_param_samples, 1, 1)
                for i in range(num_param_samples):
                    if self.mle:
                        cemb[i][mask_indices] = det_mean
                    else:
                        cemb[i][mask_indices] = det_mean + torch.randn_like(det_mean) * torch.exp(0.5 * det_logvar)

                return cemb.reshape(-1, out_shape, 1, 1)

            cemb1 = context_embed(self.contextembed1, self.c1_det_mean, self.c1_det_logvar, 2*self.n_feat)
            cemb2 = context_embed(self.contextembed2, self.c2_det_mean, self.c2_det_logvar, self.n_feat)

        else:
            c = c.repeat(num_param_samples)
            # mask out context if context_mask == 1
            context_mask = context_mask.repeat(num_param_samples)
            context_mask = context_mask[:, None]
            # convert context to one hot embedding
            c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
            context_mask = context_mask.repeat(1,self.n_classes)
            context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
            c = c * context_mask

            cemb1 = self.contextembed1(c, num_param_samples).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c, num_param_samples).view(-1, self.n_feat, 1, 1)

        # embed context, time step
        temb1 = self.timeembed1(t, num_param_samples).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t, num_param_samples).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(self.up0_ct(hiddenvec, num_param_samples))
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2, num_param_samples)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1, num_param_samples)
        out = self.out_conv1(torch.cat((up3, x), 1), num_param_samples)
        out = self.out(out)
        out = self.out_conv2(out, num_param_samples)
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, num_param_samples=10):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        out = self.nn_model(x_t, c, _ts / self.n_T, context_mask, num_param_samples)

        noise = noise.repeat(num_param_samples, 1, 1, 1)
        return self.loss_mse(noise, out)

    def sample(self, num_noise_samples, num_classes, size, device, guide_w = 0.0, num_param_samples=10, return_dataset=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(num_noise_samples, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,num_classes).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(num_noise_samples/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[num_noise_samples:] = 1. # makes second half of batch context free

        # x_i_store = [] # keep track of generated steps in case want to plot something
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(num_noise_samples,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(num_noise_samples, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask, num_param_samples).reshape(num_param_samples, -1, *size).mean(dim=0)

            # eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:num_noise_samples]
            eps2 = eps[num_noise_samples:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:num_noise_samples]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            # if i%20==0 or i==self.n_T or i<8:
            #     x_i_store.append(x_i.detach().cpu())

        if return_dataset:
            sampled_dataset = TensorDataset(torch.stack(x_i), c_i[:num_noise_samples])
            return sampled_dataset

        return x_i