import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


def stack_params(model):
    mu = []
    logvar = []

    for name, param in model.named_parameters():
        if name[-4:] == 'mean':
            mu.append(param.view(-1))
        elif name[-6:] == 'logvar':
            logvar.append(param.view(-1))

    stacked_mu, stacked_logvar = torch.cat(mu), torch.cat(logvar)
    print(f"stacked_mu device: {stacked_mu.device}")
    print(f"stacked_logvar device: {stacked_logvar.device}")
    return stacked_mu, stacked_logvar


def kld(model, prior_mu, prior_logvar):
    mu, logvar = stack_params(model)

    print(f"mu device: {mu.device}")
    print(f"logvar device: {logvar.device}")

    print(f"prior_mu device: {prior_mu.device}")
    print(f"prior_logvar device: {prior_logvar.device}")

    log_std_diff = prior_logvar - logvar
    mu_diff = (torch.exp(logvar) + (mu - prior_mu)**2) / torch.exp(prior_logvar)

    print(f"log_std_diff device: {log_std_diff.device}")
    print(f"mu_diff device: {mu_diff.device}")

    kld = 0.5 * torch.sum(log_std_diff + mu_diff - 1)
    print(f"kld device: {kld.device}")
    return kld


def train_epoch(ddpm, dataloader, optim, device, num_param_samples=10, prior_mu=None, prior_logvar=None, mle=True):
    ddpm.train()

    pbar = tqdm(dataloader)
    loss_ema = None
    for x, c in pbar:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)
        try:
            loss = ddpm(x, c, num_param_samples)
            if not mle:
                loss += (kld(ddpm, prior_mu, prior_logvar) / len(dataloader.dataset))
            loss.backward()
        except RuntimeError as e:
            print("Exception caught:")
            print(e)
            # Print the devices of all tensors involved
            print("Devices:")
            print(f"loss: {loss.device}")
            print(f"x: {x.device}")
            print(f"c: {c.device}")
            for name, param in ddpm.named_parameters():
                print(f"{name}: {param.device}")
            if not mle:
                print((kld(ddpm, prior_mu, prior_logvar) / len(dataloader.dataset)).device)
                print(f"prior_mu: {prior_mu.device}")
                print(f"prior_logvar: {prior_logvar.device}")
            raise e

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()

    return x, c


def eval(ep, ddpm, n_classes, save_dir, device, ws_test=[0.0, 0.5, 2.0], save_gif=False, num_eval_samples=10):
    ddpm.eval()
    with torch.no_grad():
        n_noise_samples = 4 * n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = ddpm.sample(n_noise_samples, n_classes, (1, 28, 28), device, guide_w=w, num_param_samples=num_eval_samples)
            grid = make_grid(x_gen*-1 + 1, nrow=10)
            save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

            if save_gif:
                fig, axs = plt.subplots(nrows=int(n_noise_samples/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    plots = []
                    for row in range(int(n_noise_samples/n_classes)):
                        for col in range(n_classes):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                            plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    return plots
                ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])
                ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
