
import torch
from torch import optim
import torch.nn as nn
from torch.nn.functional import F
import math
# import numpy as np
# from fastprogress import progress_bar
# from utils import *


class convNet(nn.Module):
  def __init__(self,in_channels,out_channels,mid_channels=None,residual=None):
    super().__init__()
    self.in_channels=in_channels
    self.residual=residual
    self.out_channels=out_channels
    if not mid_channels:
      self.mid_channels=out_channels
    self.conv1=nn.sequential(
        nn.conv(in_channels,self.mid_channels,kernel_size=3, bias=None),
        nn.GroupNorm(1,self.mid_channels),
        nn.GELU(),
        nn.conv(self.mid_channels,out_channels),
        nn.GroupNorm(1,self.out_channels),
        nn.GELU()
        

        
    )
  def forward(self,tensor):
      out=self.conv1(tensor)
      if self.residual:
        out_inter=(tensor+out)
        out=F.silu(out_inter)


class DownSample(nn.Module):
  def __init__(self,in_channels,out_channels,time_dim):
    super.__init__()
    
    self.down=nn.Sequential(
      convNet(in_channels, in_channels,residual=True),
      convNet(in_channels,out_channels),
      nn.GroupNorm(1,out_channels),
      nn.GELU(),
      nn.MaxPool2d(2,2)
      



    )
    
    self.emb=nn.Sequential(
      nn.Linear(
        time_dim,
        out_channels
      ),
      nn.SiLu()
      
    )
  def forward(self, tensor, t):
    x1=self.down(tensor)
    emb1=self.emb(t)
    out=x1+emb1
    return out


class SelfAttention(nn.Module):
  def __init__(self,channels,residual=False):
    super.__init__()
    self.residual= residual
    self.norm=nn.LayerNorm(channels)
    self.att=nn.MultiheadAttention(channels,4,vdim=channels,kdim=channels,batch_first=True)
    self.lin=nn.Sequential(
      nn.Linear(channels,channels),
      nn.SiLU()
    )
  def forward(self,tensor):
    #used residual for easy gradient propagation
    batch,channel,height,width=tensor.shape()
    new=tensor.view(batch,channel,height*width).permuate(0,2,1)
    res=new
    x_1=self.norm(new)
    emb_out,emb_wei=self.att(x_1)
    emb_out=self.lin(emb_out)
    if self.residual:

    
      emb_out=res+emb_out
      emb_out=F.gelu(emb_out)
    

    return emb_out


class upSample(nn.Module):
    def __init__(self,in_channels,out_channels,time_dim):
      super.__init__()
      self.up=nn.Upsample(scale_factor=2)
      self.con1=convNet(in_channels, in_channels,residual=True)
      self.conv2=convNet(in_channels=in_channels,out_channels=out_channels)
      self.norm=nn.GroupNorm(1,out_channels)
      self.emb=nn.Sequential(
      nn.Linear(
        time_dim,
        out_channels
      ),
      nn.SiLu()
      
    )
    def forward(self,tensor,skip_x,t):
      x1=self.up(tensor)
      x_conc=torch.cat((x1,skip_x),dim=1)
      x2=self.conv1(x_conc)
      x_out=self.conv2(x2)
      emb_x=self.emb(t)
      emb=emb_x[:,:,None,None].expand(-1,-1,x_out[-2],x_out[-1])

      # x3=self.norm(x2)
      return x_out+emb
    
class unet_cond(nn.Module):
#when initiazled it can take a boolean specifinig whether or not its conditonal ddpms or not
    def __init__(self,time_dim=256,num_classes=None,in_channels=None,):
      super.__init__()
      self.dim=time_dim
      
      self.num_classes=num_classes
      self.label_emb=nn.Embedding(num_classes,time_dim)
      self.conv1=convNet(3,64)
      
      self.down1=DownSample(64,128)
      self.att1=SelfAttention(128)
      self.down2=DownSample(128,256) #maybe residual
      self.att2=SelfAttention(256,residual=True)
      self.down3=DownSample(256,256)
      self.att3=SelfAttention(256)

      self.bot_neck1=convNet(256,512)
      self.bot_neck2=convNet(512,512)

      self.up1=upSample(512,256)
      self.att4=SelfAttention(256)
      self.up2=upSample(256,128)
      self.att5=SelfAttention(128,residual=True)
      self.up3=upSample(128,64)
      self.att6=SelfAttention(64)
      self.con2=convNet(64,3)



      
      
    
     
      # if conditional:
      #   pass
    def timeLabelEncode(self,t,label=None):
      half=self.dim/2
      freqs=torch.exp(-math.log(10000)*torch.arange(0,half)/half)
      sins=torch.sin(t*freqs)
      cos_=torch.cos(t*freqs)
      emb=torch.cat([sins,cos_],dim=-1)
      if self.dim % 2 == 1:  # pad if odd requested dimension
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1)], dim=-1)
      
      if label:
        lab_emb1=self.label_emb(label)

        emb+=lab_emb1
      return emb
    

    def forward(self,x,time,label=None):
        
        t=self.timeLabelEncode(time,label)
        x1 = self.conv1(x)
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)
        x4 = self.down3(x3, t)
        x4 = self.att3(x4)

        x4 = self.bot_neck1(x4)
        
        x4 = self.bot_neck2(x4)
        #this is where the global skipping happens, where the ouputs of the encder at one step in directly fed into the corrsponding step in the decoder, in this case x3 came direclty form above during the downattmpling
        x = self.up1(x4, x3, t)
        x = self.att4(x)
        x = self.up2(x, x2, t)
        x = self.att5(x)
        x = self.up3(x, x1, t)
        x = self.att6(x)
        output = self.conv2(x)
        return output        



    #accepts an image, and a time step scalar whihc we'll have to encode
   
    
    

