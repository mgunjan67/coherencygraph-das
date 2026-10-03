"""Same recurrence family, separately fitted for the pick-free replication."""
import numpy as np
import torch
from torch import nn
from .models import ModernBackbone
from .causal_validation import Q


class CausalSpectralModel(nn.Module):
    def __init__(self,lags):
        super().__init__()
        self.encoder=ModernBackbone(135,96,2,.1,'bissm_psd')
        self.head=nn.Linear(96,4*257)
        basis=np.exp(1j*np.asarray(lags)[:,None]*Q)
        self.register_buffer('basis_re',torch.tensor(basis.real,dtype=torch.float32))
        self.register_buffer('basis_im',torch.tensor(basis.imag,dtype=torch.float32))

    def forward(self,x):
        p=torch.softmax(self.head(self.encoder(x)).reshape(-1,8,4,257),-1)
        y=torch.stack([torch.einsum('nxbq,lq->nxbl',p,self.basis_re),
                       torch.einsum('nxbq,lq->nxbl',p,self.basis_im)],-1)
        return y,p
