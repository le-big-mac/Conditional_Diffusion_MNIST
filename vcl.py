import torch
from torch.utils import data

from mnist import get_split_MNIST
from model import ContextUnet, DDPM
from utils import eval, kld, stack_params, train_epoch

n_epoch = 20
batch_size = 256
n_T = 400 # 500
device = "cuda:0"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = False
save_dir = './data/diffusion_outputs10/'
mle_comp = False

digit_datasets = get_split_MNIST()

# MLE training
nn_model = ContextUnet(1, n_feat, n_classes, mle=True)
ddpm = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
zero_loader = data.DataLoader(digit_datasets[0], batch_size=batch_size, shuffle=True, num_workers=5)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
prior_mu, prior_logvar = stack_params(nn_model)

for ep in range(n_epoch):
    print(f"Epoch {ep}")
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    train_epoch(ddpm, zero_loader, optim, device)

nn_model = ContextUnet(1, n_feat, n_classes, mle=False)
ddpm = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

for digit in range(n_classes):
    digit_data = digit_datasets[digit]
    digit_loader = data.DataLoader(digit_data, batch_size=batch_size, shuffle=True, num_workers=5)

    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        x, c = train_epoch(ddpm, digit_loader, optim, device, prior_mu, prior_logvar, mle=mle_comp)
        save_gif = True if ep == n_epoch - 1 or ep%5 == 0 else False
        eval(ep, ddpm, x, c, n_classes, save_dir, device, save_gif=save_gif)

    prior_mu, prior_logvar = stack_params(nn_model)
