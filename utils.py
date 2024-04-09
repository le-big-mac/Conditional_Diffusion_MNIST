# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import math
import pickle
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model.model import ContextUnet, DDPM
from data.mnist import merge_datasets

def stack_params(model):
    mu = []
    logvar = []

    for name, param in model.named_parameters():
        if name[-4:] == 'mean':
            mu.append(param.view(-1))
        elif name[-6:] == 'logvar':
            logvar.append(param.view(-1))

    stacked_mu, stacked_logvar = torch.cat(mu), torch.cat(logvar)
    return stacked_mu, stacked_logvar


def kld(model, prior_mu, prior_logvar):
    mu, logvar = stack_params(model)

    log_std_diff = prior_logvar - logvar
    mu_diff = (torch.exp(logvar) + (mu - prior_mu)**2) / torch.exp(prior_logvar)

    return 0.5 * torch.sum(log_std_diff + mu_diff - 1)


def model_setup(hparams, mle, n_T, device):
    ddpm = DDPM(ContextUnet(1, hparams["n_feat"], hparams["n_classes"],
                            logvar_init=hparams["logvar_init"], mle=mle),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    return ddpm


def train_epoch(ddpm, dataloader, optim, device, num_param_samples=10, prior_mu=None, prior_logvar=None, mle=True):
    ddpm.train()

    pbar = tqdm(dataloader)
    loss_ema = None
    kl_ema = None
    mse_ema = None
    for x, c in pbar:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)

        loss = ddpm(x, c, num_param_samples)
        mse_val = loss.item()
        if not mle:
            kl = kld(ddpm, prior_mu, prior_logvar) / len(dataloader.dataset)
            loss += kl
            kl = kl.item()
        loss.backward()
        optim.step()

        if loss_ema is None:
            loss_ema = loss.item()
            if not mle:
                mse_ema = mse_val
                kl_ema = kl
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            if not mle:
                mse_ema = 0.95 * mse_ema + 0.05 * mse_val
                kl_ema = 0.95 * kl_ema + 0.05 * kl
        if mle:
            pbar.set_description(f"loss: {loss_ema:.4f}")
        else:
            pbar.set_description(f"loss: {loss_ema:.4f} kl: {kl_ema:.4f} mle: {mse_ema:.4f}")


@torch.no_grad()
def eval(ddpm, n_classes, save_loc, device, ws_test=[2.0], num_param_samples=10):
    ddpm.eval()
    n_noise_samples = 4 * n_classes
    for w in ws_test:
        x_gen = ddpm.sample(n_noise_samples, n_classes, (1, 28, 28), device, guide_w=w, num_param_samples=num_param_samples)
        grid = make_grid(x_gen, nrow=n_classes)
        save_image(grid, f"{save_loc}_w{w}.png")
        print(f"saved image at {save_loc}_w{w}.png")


@torch.no_grad()
def sample_dataset(ddpm, n_classes, save_dir, device, w, num_datapoints=200, num_param_samples=10):
    ddpm.eval()

    samples = []
    for i in range(math.ceil(num_datapoints // 200)):
        num_samples = n_classes * (200 // n_classes)
        sampled_dataset = ddpm.sample(num_samples, n_classes, (1, 28, 28), device, guide_w=w, num_param_samples=num_param_samples, return_dataset=True)
        samples.append(sampled_dataset)

    sampled_dataset = merge_datasets(samples)
    with open(f"{save_dir}/{n_classes-1}.pkl", "wb+") as f:
        pickle.dump(sampled_dataset, f)
