
import torch
from torch import nn
from torch.nn import functional as F
import math
       
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()   
        self.size = d_model
        self.eps = eps

        '''Create two learnable parameters to calibrate normalisation'''
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

class HyperLinear(nn.Module):
    def __init__(self, hyper_dm, input_dm, output_dm=None,hyper=True):
        super().__init__()
        self.hyper = hyper
        if output_dm is None:   output_dm = input_dm
        self.U  = nn.Linear(input_dm,output_dm)
        if self.hyper:
            self.Wu = nn.Linear(hyper_dm,output_dm)
            self.Wb = nn.Linear(hyper_dm,output_dm)
        
    def forward(self, x, xh=None):  
        return self.U(x) if self.hyper==False else self.U(x) * self.Wu(xh) + self.Wb(xh)
    
class QLinear(nn.Module):
    def __init__(self, dim, hyper_dim, childattention):
        super().__init__()
        self.childattention = childattention
        if childattention=='self':          self.q_linear = HyperLinear(hyper_dim // 2, dim // 2, dim // 2, hyper= False)
        if childattention=='hyperself':     self.q_linear = HyperLinear(hyper_dim // 2, dim // 2)
        if childattention=='hypercross':    self.q_linear = HyperLinear(hyper_dim // 2, hyper_dim // 2, dim // 2, hyper= False)
    
    def forward(self, hlr, xh):   
        if self.childattention in ['self','hyperself']: return self.q_linear(hlr,xh)
        if self.childattention=='hypercross':   return self.q_linear(xh.repeat(1,2,1))
            
class KLinear(nn.Module):
    def __init__(self, dim, hyper_dim, childattention):
        super().__init__()
        self.childattention = childattention
        self.k_linear = HyperLinear(hyper_dim // 2, dim // 2, hyper = childattention!='self')
    
    def forward(self, hlr, xh):   
        return self.k_linear(hlr,xh)
        
class ChildHAggregation(nn.Module):
    def __init__(self, dim, idim, hyper_dim=None, hyper=False, dropout=0.1, childattention=None):
        super().__init__()
        self.hyper = hyper
        self.childattention = childattention 

        self.hidden = HyperLinear(hyper_dim // 2, dim, dim // 2, hyper)
        self.leaf = HyperLinear(hyper_dim // 2, idim, dim // 2, hyper)

        if self.childattention is not None:
            if hyper==False:    self.childattention = 'self'
            self.q_linear = QLinear(dim, hyper_dim, self.childattention)
            self.k_linear = KLinear(dim, hyper_dim, self.childattention)
            self.norm = Norm(dim)
        
    '''hl, hr : (batch, dim//2), xh: (batch, hyper_dim //2), xw: (batch, idim)'''
    def forward(self, hl, hr, xw, xh=None):
        x = torch.concatenate([hl,hr],axis=1)     # (batch,dim)
            
        if self.childattention is not None:
            hlr = torch.stack([hl,hr],1)    # (batch, 2, dim//2)
            q = self.q_linear(hlr, xh.unsqueeze(1) if xh is not None else None)      # (batch, 2, dim//2)
            k = self.k_linear(hlr, xh.unsqueeze(1) if xh is not None else None)      # (batch, 2, dim//2)

            scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(hl.size(1))    # (batch, 2, 2)
            scores = F.softmax(scores, dim=-1)                              # (batch,2,2)
            x += torch.matmul(scores, hlr).view(hl.size(0),-1)              # (batch,dim)
            # x += self.dropout(torch.matmul(scores, hlr).view(bs,-1))       # (batch,dim)

            x = self.norm(x)

        return self.hidden(x,xh)+self.leaf(xw,xh)


class TreeLSTM(nn.Module):
    # def __init__(self, dim, leaf_dim, hyper_dim = None, hyper=False, mode='childsum', dropout=0.1):
    def __init__(self, dim, leaf_dim, hyper_dim = None, hyper=False, childattention=None, dropout=0.1):

        super().__init__()
        assert (dim % 2) == 0
        
        self.gh  = ChildHAggregation(dim, leaf_dim, hyper_dim, hyper, dropout, childattention) #=True if mode=='childattention' else False)
        self.ih  = ChildHAggregation(dim, leaf_dim, hyper_dim, hyper, dropout, childattention) #=True if mode=='childattention' else False)
        self.flh = ChildHAggregation(dim, leaf_dim, hyper_dim, hyper, dropout, childattention) #=True if mode=='childattention' else False)
        self.frh = ChildHAggregation(dim, leaf_dim, hyper_dim, hyper, dropout, childattention) #=True if mode=='childattention' else False)
        self.oh  = ChildHAggregation(dim, leaf_dim, hyper_dim, hyper, dropout, childattention) #=True if mode=='childattention' else False)
                
    '''xl,xr: (batch, dim), xh: (batch, hyper_dim)'''
    def forward(self, x_left, x_right, x_leaf, xh=None):
        '''Split x to h and c and construct hlr by concatenating hl and hr '''
        hl,cl = torch.chunk(x_left, 2, 1)               # (batch, dim//2)
        hr,cr = torch.chunk(x_right, 2, 1)               # (batch, dim//2)
        hh,ch = torch.chunk(xh, 2, 1) if xh is not None else (None,None)    # (batch, hyper_dim//2)
        
        '''generate g,i,o,fl,fr'''
        g  = torch.tanh(self.gh(hl, hr, x_leaf, hh))           # (batch, dim//2)
        i  = torch.sigmoid(self.ih(hl, hr, x_leaf, hh))        # (batch, dim//2)
        fl = torch.sigmoid(self.flh(hl, hr, x_leaf, hh))       # (batch, dim//2)
        fr = torch.sigmoid(self.frh(hl, hr, x_leaf, hh))       # (batch, dim//2)
        o  = torch.sigmoid(self.oh(hl, hr, x_leaf, hh))        # (batch, dim//2)
        
        '''generate cp, hp and return after concatenating it together'''
        cp = i * g + fl * cl + fr * cr              # (batch, dim//2)
        hp = o * (torch.tanh(cp))                   # (batch, dim//2)
        return torch.concatenate([hp,cp],axis=1)    # (batch, dim)

    
class Linear(nn.Module):
    def __init__(self, leaf_dim, dim):
        super().__init__()
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.leaf = nn.Linear(leaf_dim, dim)
        
    def forward(self, x_left, x_right, x_leaf, xh=None):
        return self.left(x_left) + self.right(x_right) + self.leaf(x_leaf)
    
class Sum(nn.Module):
    def __init__(self, leaf_dim, dim):
        super().__init__()
        self.leaf = nn.Linear(leaf_dim, dim, bias=False)
        
    def forward(self, x_left, x_right, x_leaf, xh=None):
        return (x_left + x_right) + self.leaf(x_leaf)
        # return (xl + xr)
    
class Mean(nn.Module):
    def __init__(self, leaf_dim, dim):
        super().__init__()
        self.leaf = nn.Linear(leaf_dim, dim, bias=False)
        
    def forward(self, x_left, x_right, x_leaf, xh=None):        
        return (x_left + x_right) / 2 + self.leaf(x_leaf)
        
class Composition(nn.Module):
    def __init__(self, dim, leaf_dim, compose, hyper=False, hyper_dim = 1, dropout=0.1):
        super().__init__()
        
        self.compose = compose
        self.leaf_dim = leaf_dim
        self.dim = dim
        
        if compose=='htlstm':
            self.compose_fn = TreeLSTM(dim = dim, leaf_dim=leaf_dim, hyper_dim=hyper_dim, hyper=hyper, childattention=None)
        elif compose=='hatlstm':
            self.compose_fn = TreeLSTM(dim = dim, leaf_dim=leaf_dim, hyper_dim=hyper_dim, hyper=hyper, childattention='self', dropout= dropout)
        elif compose=='hatlstm-hyperself':
            self.compose_fn = TreeLSTM(dim = dim, leaf_dim=leaf_dim, hyper_dim=hyper_dim, hyper=hyper, childattention='hyperself', dropout= dropout)
        elif compose=='hatlstm-hypercross':
            self.compose_fn = TreeLSTM(dim = dim, leaf_dim=leaf_dim, hyper_dim=hyper_dim, hyper=hyper, childattention='hypercross', dropout= dropout)
        elif compose=='linear':
            self.compose_fn = Linear(leaf_dim=leaf_dim, dim=dim)
        elif compose=='mean':
            self.compose_fn = Mean(leaf_dim=leaf_dim, dim=dim)
        elif compose=='sum':
            self.compose_fn = Sum(leaf_dim=leaf_dim, dim=dim)
        else:
            raise(Exception('compose option not identified'))
            
    def forward(self, x_left=None, x_right=None, x_leaf=None, xh=None):
        bs = x_leaf.size(0) if x_leaf is not None else x_left.size(0)
            
        if x_leaf is None:  x_leaf  = torch.zeros(bs,self.leaf_dim).to(next(self.parameters()).device)  
        if x_left is None:  x_left  = torch.zeros(bs,self.dim).to(next(self.parameters()).device)  
        if x_right is None: x_right = torch.zeros(bs,self.dim).to(next(self.parameters()).device) 
        
        return self.compose_fn(x_left=x_left, x_right=x_right, x_leaf=x_leaf, xh=xh)
        