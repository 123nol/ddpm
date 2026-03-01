import torch
import math
from module import unet_cond
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext


from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
from utils.py import *

class Diffusion:
    def __init__(self, num_classes, in_channels=1, timesteps=1000, img_size=256, ddim_timesteps=50, device="cuda"):
        self.beta, self.abar = self.cosine_beta_schedule_torch(timesteps, device=device)
        self.model = UNetDDPM(in_channels=in_channels, num_classes=num_classes)
        self.timesteps = timesteps
        self.img_size = img_size
        self.in_channels = in_channels
        self.ddim_timesteps = ddim_timesteps
        self.device = device


    @staticmethod
    def cosine_beta_schedule_torch(T, s=0.008, max_beta=0.999, device=None, dtype=torch.float32):
        t = torch.linspace(0, T, T + 1, device=device, dtype=torch.float64)
        f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        abar = f / f[0]
        alphas = (abar[1:] / abar[:-1]).to(dtype)
        betas = (1.0 - alphas).clamp_(min=0.0, max=max_beta)
        return betas, abar.to(dtype)

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.timesteps, size=(n,), device=self.device)

    def noise_image(self, x_0, t):
        sqrt_abar = torch.sqrt(self.abar[t])[:, None, None, None]
        sqrt_one_minus_abar = torch.sqrt(1 - self.abar[t])[:, None, None, None]
        eps = torch.randn_like(x_0)
        return sqrt_abar * x_0 + sqrt_one_minus_abar * eps, eps


    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_image(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)

            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()



    @torch.no_grad()
    def ddim_indices_torch(self, T, N, method="time", device=None, dtype=torch.float64):
        # Ensure we use the correct device
        if device is None:
            device = self.device

        if method == "time":
            tau = torch.round(torch.linspace(0, T, N + 1, device=device)).to(torch.long)
            return torch.unique(tau)

        # --- log-SNR method ---
        eps = torch.finfo(dtype).eps
        # Use self.abar which is already on the correct device
        logsnr = torch.log(self.abar / (1.0 - self.abar + eps))

        logsnr_rev = torch.flip(logsnr, dims=[0])
        # Crucial fix: ensure grid is on the same device as logsnr_rev
        grid = torch.linspace(logsnr[0].item(), logsnr[-1].item(), N + 1, device=device, dtype=dtype)
        grid_rev = torch.flip(grid, dims=[0])

        idx_rev = torch.searchsorted(logsnr_rev, grid_rev, right=False)
        tau = (T - idx_rev).clamp(min=0, max=T).to(torch.long)
        tau, _ = torch.sort(tau)
        return torch.unique(tau)

    @torch.no_grad()
    def ddim_sample_with_intermediates(self, label, guide_scale=3., capture_interval=10, N=50):
        self.model.eval()
        batch_size = label.shape[0]

        # 1. Correct Shape: (Batch, Channels, Height, Width)
        x = torch.randn((batch_size, self.in_channels, self.img_size, self.img_size), device=self.device)

        # 2. Get DDIM schedule
        tau = self.ddim_indices_torch(self.timesteps, N, method="time")
        # Reverse tau for sampling: [T, ..., 0]
        tau = torch.flip(tau, dims=[0])

        intermediates = []

        for i in range(len(tau) - 1):
            t_now = tau[i]
            t_next = tau[i+1]

            # Create tensors for model input
            t_tensor = torch.full((batch_size,), t_now, device=self.device, dtype=torch.long)

            # 3. Model Prediction (Guidance)
            pre_pred_noise = self.model(x, t_tensor, label)
            uncond_pred = self.model(x, t_tensor, None)
            pred_noise = torch.lerp(uncond_pred, pre_pred_noise, guide_scale)

            # 4. DDIM Logic
            abar_now = self.abar[t_now].view(-1, 1, 1, 1)
            abar_next = self.abar[t_next].view(-1, 1, 1, 1)

            # Predict x0
            x0_hat = (x - torch.sqrt(1 - abar_now) * pred_noise) / torch.sqrt(abar_now)

            # Compute x_next
            x = torch.sqrt(abar_next) * x0_hat + torch.sqrt(1 - abar_next) * pred_noise

            # 5. Capture snapshot
            if i % capture_interval == 0 or i == len(tau) - 2:
                # Normalize and move to CPU for plotting
                out = (x.clamp(-1, 1) + 1) / 2
                out = (out * 255).type(torch.uint8)
                intermediates.append(out.cpu())

        return intermediates





    @torch.inference_mode()
    def ddpm_sample_with_intermediates(self, label, guide_scale=3., capture_interval=100):
        self.model.eval()
        # Shape must be (Batch, Channels, Height, Width)
        x = torch.randn((label.shape[0], self.in_channels, self.img_size, self.img_size), device=self.device)
        intermediates = []

        for i in reversed(range(0, self.timesteps)):
            # Create a batch of timesteps as Tensors
            t_tensor = torch.full((label.shape[0],), i, device=self.device, dtype=torch.long)

            # Classifier-Free Guidance
            pre_pred_noise = self.model(x, t_tensor, label)
            uncond_pred = self.model(x, t_tensor, None)
            pred_noise = torch.lerp(uncond_pred, pre_pred_noise, guide_scale)

            # Get constants for current t and reshape for broadcasting
            beta = self.beta[i].view(-1, 1, 1, 1)
            abar = self.abar[i].view(-1, 1, 1, 1)
            abar_next = self.abar[i+1].view(-1, 1, 1, 1)

            noise = torch.randn_like(x) if i > 0 else 0

            # DDPM Step
            # Formula: x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-abar) * eps) + sigma*z
            alpha = 1 - beta
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - abar_next)) * pred_noise) + torch.sqrt(beta) * noise

            if i % capture_interval == 0 or i == 0:
                # Normalize to 0-255 for visualization
                out = (x.clamp(-1, 1) + 1) / 2
                out = (out * 255).type(torch.uint8)
                intermediates.append(out.cpu())

        return intermediates

    def train_step(self, loss):
          self.optimizer.zero_grad()
          self.scaler.scale(loss).backward()
          self.scaler.step(self.optimizer)
          self.scaler.update()

          self.scheduler.step()




    def one_epoch(self, train=True):
          avg_loss = 0.
          if train: self.model.train()
          else: self.model.eval()
          pbar = progress_bar(self.train_dataloader, leave=False)
          for i, (images, labels) in enumerate(pbar):
              with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                  images = images.to(self.device)
                  labels = labels.to(self.device)
                  t = self.sample_timesteps(images.shape[0]).to(self.device)
                  x_t, noise = self.noise_image(images, t)
                  if np.random.random() < 0.1:
                      labels = None
                  predicted_noise = self.model(x_t, t, labels)
                  loss = self.mse(noise, predicted_noise)
                  avg_loss += loss
              if train:
                  self.train_step(loss)

              pbar.comment = f"MSE={loss.item():2.3f}"
          return avg_loss.mean().item()


    def load(self, path):
        # If the path passed is a directory, look for ckpt.pt inside it
        if os.path.isdir(path):
            path = os.path.join(path, "ckpt.pt")

        # Load the state dict and map it to the correct device (cpu/cuda)
        state_dict = torch.load(path, map_location=self.device)

        # Load into the model
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully from {path}")


    def save_model(self, run_name, epoch=-1):

          torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))

          torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))


    def prepare(self, args):
          mk_folders(args.run_name)
          self.train_dataloader, self.val_dataloader = get_alphabet(args)
          self.model.to(self.device) # Add this line to move the model to the correct device
          self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
          self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                  steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
          self.mse = nn.MSELoss()

          self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
          for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
              logging.info(f"Starting epoch {epoch}:")
              _  = self.one_epoch(train=True)

              ## validation
              if args.do_validation:
                  avg_loss = self.one_epoch(train=False)

          # save model
          self.save_model(run_name=args.run_name, epoch=epoch)
