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

# from utils import *



class Diffusion:
  def __init__(self,num_classes,in_channels=3,timesteps=1000,img_size=256,ddim_timesteps=50,device="cuda"):
    self.beta,self.abar=self.cosine_beta_schedule_torch(timesteps)
    self.model=unet_cond(num_classes,in_channels=in_channels)
    self.timesteps=timesteps
    self.img_size=img_size
    self.in_channels=in_channels
    self.ddim_timesteps=ddim_timesteps
    self.device = device
    
    

  def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,))

  @torch.no_grad
  def cosine_beta_schedule_torch(T: int, s: float = 0.008, max_beta: float = 0.999, device=None, dtype=torch.float32):
      t = torch.linspace(0, T, T + 1, device=device, dtype=torch.float64) 
      f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
      abar = f / f[0]
      alphas = (abar[1:] / abar[:-1]).to(dtype)
      betas = (1.0 - alphas).clamp_(min=0.0, max=max_beta)
      return betas,abar

  def noise_image(self,x_0,t):
        
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
                x_t, noise = self.noise_images(images, t)
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
  def ddim_indices_torch(
    self,
    T: int = 1000,
    N: int = 50,
    method: str = "time",   
    
    device=None,
    dtype=torch.float64,
):
 

   

    if method == "time":
        tau = torch.round(torch.linspace(0, T, N + 1, device=device)).to(torch.long)
        tau = torch.unique(tau)  # just in case
        return tau

    # --- uniform in log-SNR ---
    eps = torch.finfo(dtype).eps
    logsnr = torch.log(self.abar / (1.0 - self.abar + eps))  # (T+1,), typically decreasing in t

    # searchsorted requires ascending arrays → flip both to ascending
    logsnr_rev = torch.flip(logsnr, dims=[0])  # ascending
    grid = torch.linspace(logsnr[0].item(), logsnr[-1].item(), N + 1, device=device, dtype=dtype)
    grid_rev = torch.flip(grid, dims=[0])      # ascending to match logsnr_rev

    # indices in the reversed domain
    idx_rev = torch.searchsorted(logsnr_rev, grid_rev, right=False)  # (N+1,)
    # map back to original indices
    tau = (T - idx_rev).clamp(min=0, max=T).to(torch.long)
    tau, _ = torch.sort(tau)
    tau = torch.unique(tau)  # ensure strictly non-decreasing unique steps
    return tau
  

  @torch.no_grad
  def ddim_sample_with_intermediates(self,label,guide_scale=3.,capture_interval=100,N=50):
      model=self.model
      intermeidate=[]
      indices=self.ddim_indices_torch(self.timesteps,self.ddim_timesteps,"log")
      abars=self.abar[indices]
      betas=self.abar[indices]
      x=torch.randn((self.img_size,self.img_size,self.in_channels),dtype=float)
      intermeidate=[]

      for t in range(self.ddim_timesteps,1,-1):
        pre_pred_noise=model(x,t,label)
        uncond_pred=model(x,t,None)
        abar_prev=abars[t][:,None,None]
        abar_cur=abars[t-1][:,None,None]

        pred_noise=(guide_scale*pre_pred_noise)+((1-guide_scale)*uncond_pred)

        x0_hat = (x - torch.sqrt(1 - abar_prev) * pred_noise) / torch.sqrt(abar_prev)

        x = torch.sqrt(abar_cur) * x0_hat + torch.sqrt(1 - abar_cur) * pred_noise
        
        if t%capture_interval==0:
          x=(x.clamp(-1,1))+1
          x=(x*255).to(torch.uint)
          intermeidate.append(x.cpu)
      return intermeidate

         
     
     
    

  
  @torch.inference_mode()
  def ddpm_sample_with_intermediates(self,
                                
                                label: int,
                                guide_scale: float = 3.,
                                capture_interval: int = 100
                                ):
      model=self.model
      intermeidate=[]
      x=torch.randn((self.img_size,self.img_size,self.in_channels),dtype=float)
      for t in range(self.timesteps,1,-1):
          pre_pred_noise=model(x,t,label)
          uncond_pred=model(x,t,None)
          pred_noise=(guide_scale*pre_pred_noise)+((1-guide_scale)*uncond_pred)
          
          beta=self.beta[t][:,None,None]
          abar=self.abar[t][:,None,None]
          noise=torch.randn_like(x) if t>1 else torch.zeros_like(x)
          post_mean= (1/(1-beta).sqrt()) * (x - (beta/ (1-abar).sqrt()) * pred_noise)
          post_var=beta.sqrt()
          x = post_mean + post_var * noise
      if t%capture_interval==0:
          x=(x.clamp(-1,1))+1
          x=(x*255).to(torch.uint)

          intermeidate.append(x.cpu)
      return intermeidate
      
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
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
               
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()



  def load(self, model_cpkt_path, model_ckpt="ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        

  def save_model(self, run_name, epoch=-1):
        
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        

  def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
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

