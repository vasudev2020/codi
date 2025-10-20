from transformers import PretrainedConfig, PreTrainedModel
from transformers import RobertaTokenizer

import torch
from torch import nn

from Compositions import Composition
from Decoder import Decoder

from Loader import DataLoader
l = DataLoader(100)

class ValidityPrediction(nn.Module):
    def __init__(self, dim, h_dim=100):
        super(ValidityPrediction,self).__init__()
        # self.linear1 = nn.Linear(dim,h_dim)
        # self.linear2 = nn.Linear(h_dim,2)
        self.linear = nn.Linear(dim,2)
        # self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    ''' X: FloatTensor(N,neg+1,dm), y: LongTensor(N,neg+1) -> loss (mean): scalar '''
    def forward(self, X, y):
        # h = self.linear1(X)
        # h = self.relu(h)
        # o = self.linear2(h)
        o = self.linear(X)
        # y = y.masked_fill(mask == 0, -100) # ignore loss from masked node
        return self.criterion(o.view(-1,2),y.view(-1))
       
class CategoryPrediction(nn.Module):
    def __init__(self, C, dim, h_dim=100):
        super(CategoryPrediction,self).__init__()
        self.linear1 = nn.Linear(dim,h_dim)
        self.linear2 = nn.Linear(h_dim,C)
        self.C = C
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100,reduction='none')
        
    ''' X: FloatTensor(N,dx), y: LongTensor(N) -> loss (mean): scalar '''
    def forward(self, X, y, weight=None):
        if weight is None:  weight = torch.ones(y.size())
        weight = weight.to(next(self.parameters()).device)
        # print(weight)
        h = self.linear1(X)
        h = self.relu(h)
        o = self.linear2(h)
        
        # y = y.masked_fill(mask == 0, -100) # ignore loss from masked node
        # return self.criterion(o.view(-1,self.C),y.contiguous().view(-1)) 
        return torch.mul(self.criterion(o.view(-1,self.C),y.contiguous().view(-1)), weight).sum() / weight.sum()
    
class CoDiConfig(PretrainedConfig):
    def __init__(self, 
                 max_str_len: int = 100,            # Maximum number of tokens generated for auxiliary tasks
                 max_nodes_per_batch: int = 128,    # Maximum number of nodes per batch considering for text generation auxiliary tasks
                 init_dim: int = 300,               # Dimension of initializing embedding (300 for GloVe)
                 C: int = 100,                      # Maximum number of syntactic category
                 dim: int = 100,                     # Dimension of semantic embedding
                 dx: int = 50,                      # Dimension of syntactic embedding
                 compose: str = 'hatlstm',          # Name of the compositional operator
                 neg_samples: int = 10,             # Number of negative samples per positive sample for sem val prediction
                 
                 dropout: float = 0.1,              # Dropout rate
                 weighted_aux = False,              # Weighted auxiliary task loss for nodes 
                 aux_mask = '111111',
                 **kwargs,):

        super().__init__(**kwargs)
        
        self.max_str_len = max_str_len
        self.max_nodes_per_batch = max_nodes_per_batch
        self.init_dim = init_dim
        self.C = C 
        self.dim = dim 
        self.dx = dx 
        self.compose = compose 
        self.neg_samples = neg_samples
        
        self.dropout = dropout 
        self.weighted_aux = weighted_aux
        self.aux_mask = aux_mask
        
class CoDiModel(PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)        
        tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

        # self.num_aux_tasks =  8 # 6 #
        self.num_aux_tasks =  6 if config.neg_samples==0 else 8
        self.weight = nn.Parameter(torch.ones(self.num_aux_tasks), requires_grad=True)

        self.syntactic_embedding = nn.Embedding(config.C,config.dx)
        
        self.syntactic_compose = Composition(dim = config.dx, leaf_dim = config.dx, compose = config.compose, hyper = False,dropout = config.dropout) 
        self.semantic_compose  = Composition(dim = config.dim, leaf_dim = config.init_dim, compose = config.compose, hyper = True, hyper_dim = config.dx, dropout = config.dropout)
        
        self.SynPosPred  = CategoryPrediction(2,config.dx)
        self.SemPosPred  = CategoryPrediction(2,config.dim)

        # if self.num_aux_tasks == 8:
        self.SynValPred = ValidityPrediction(config.dx)
        self.SemValPred = ValidityPrediction(config.dim)

        # self.SelfStrPred  = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch)
        # self.SibStrPred   = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch)
        # self.SelfStrPred  = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch,dim_emb=2000, dim_h=2000)
        # self.SibStrPred   = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch,dim_emb=2000, dim_h=2000)
        self.SelfStrPred  = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch,dim_emb=config.dim, dim_h=config.dim)
        self.SibStrPred   = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch,dim_emb=config.dim, dim_h=config.dim)
        # self.LeftStrPred  = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch)
        # self.RightStrPred = Decoder(vocab_size = len(tokenizer), bos_token_id = tokenizer.bos_token_id, pad_token_id = tokenizer.pad_token_id, dim_z = config.dim, dropout = config.dropout, max_len = config.max_str_len, max_nodes_per_batch = config.max_nodes_per_batch)
        
        self.SelfCatPred  = CategoryPrediction(config.C,config.dx)
        self.SibCatPred  = CategoryPrediction(config.C,config.dx)
        # self.LeftCatPred  = CategoryPrediction(config.C,config.dx)
        # self.RightCatPred = CategoryPrediction(config.C,config.dx)

        self.neg_samples=config.neg_samples

        self.dim     = config.dim
        self.dx      = config.dx
        self.compose = config.compose
        self.C       = config.C
        self.weighted_aux = config.weighted_aux
        self.aux_mask = torch.LongTensor([int(d) for d in config.aux_mask]+[1,1])[:self.num_aux_tasks] 

        self.syn_noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(self.dx), torch.eye(self.dx))
        self.sem_noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

        # self.aux_mask = torch.LongTensor([int(d) for d in str(f'{config.aux_mask:06b}')])
                
    def resetAuxWeight(self):
        self.weight.data.fill_(1.0)
        
    def printParamCount(self):
        print("Number of parameters in syntactic composition:",sum(p.numel() for p in self.syntactic_compose.parameters() if p.requires_grad))
        print("Number of parameters in semantic composition:",sum(p.numel() for p in self.semantic_compose.parameters() if p.requires_grad))
        
       
    def forward(self, 
                adj_list,       # LongTensor(N , 2) - Indices of left child and right child of each node
                node_order,     # LongTensor(N) - Order of nodes for the recursive composition - 0 should be computed first then 1 and then 2 and so on.
                sem_embs,       # FloatTensr(N, dm) - All node with node_order 0 should be initialized
                syn_cats,       # LongTensor(N,3)  - Category labels of all nodes (left,self,right) - values will be in range (0,C)
                strings,        # LongTensor(N, 3, max_str_len) - Tokens indices of text content of all nodes (left, self, right) - values will be in range (0, vocab size)
                root_index,     # LongTensor(batch_size) - List of indices of root nodes - values will be in range (0,N)
                aux_tasks=True,
                ):
        
        device = next(self.parameters()).device # Retrive device name of the model which is currently loaded in
        syntactic_cats = syn_cats.to(device)
        syntactic_embs_init = self.syntactic_embedding(syntactic_cats[:,1])
        semantic_embs_init  = sem_embs.to(device)
        
        semantic_embs  = torch.zeros(semantic_embs_init.size()[0],self.dim).to(device)
        syntactic_embs = torch.zeros(syntactic_embs_init.size()[0],self.dx).to(device)
        
        for iteration in range(node_order.max() + 1):
            node_mask = node_order == iteration
            if iteration==0:
                syntactic_embs[node_mask, :] = self.syntactic_compose(x_leaf = syntactic_embs_init[node_mask, :]) 
                semantic_embs[node_mask, :]  = self.semantic_compose(x_leaf = semantic_embs_init[node_mask, :], xh = syntactic_embs[node_mask, :]) 
                continue
            left_indexes = adj_list[node_mask, 0]
            right_indexes = adj_list[node_mask, 1]
            
            '''Apply composition operation'''
            syntactic_embs[node_mask, :] = self.syntactic_compose(x_left = syntactic_embs[left_indexes,:], x_right = syntactic_embs[right_indexes,:])                
            semantic_embs[node_mask, :] = self.semantic_compose(x_left = semantic_embs[left_indexes,:], x_right = semantic_embs[right_indexes,:], xh = syntactic_embs[node_mask, :]) 
         
        if aux_tasks:
            wt = node_order+1 if self.weighted_aux else None
            
            left_index = adj_list[adj_list[:,0]!=adj_list[:,1]][:,0].unique()
            right_index = adj_list[adj_list[:,0]!=adj_list[:,1]][:,1].unique()
            position = torch.zeros(adj_list.size(0), dtype = torch.long, device = device)
            position[right_index]=1         # Mark position of each node: 0 for left and 1 for right - size (bs) 
            assert (position[left_index]==0).all()
            assert (syntactic_cats.gather(1,((position==1)*2).view(-1,1)).squeeze()==l.cats.index('NULL')).all()

            SynPosPredLoss = self.SynPosPred(syntactic_embs,position)
            SemPosPredLoss = self.SemPosPred(semantic_embs,position)

            SelfCatPredLoss  = self.SelfCatPred(syntactic_embs, syntactic_cats[:,1],weight=wt)
            SibCatPredLoss   = self.SibCatPred(syntactic_embs, syntactic_cats.gather(1,((position==0)*2).view(-1,1)).squeeze(), weight=wt) 
            # LeftCatPredLoss  = self.LeftCatPred(syntactic_embs, syntactic_cats[:,0],weight=wt)
            # RightCatPredLoss = self.RightCatPred(syntactic_embs, syntactic_cats[:,2],weight=wt)

            sib_str_index = ((position==0)*2).view(-1,1,1).repeat(1,1,strings.size(2))
            SelfStrPredLoss  = self.SelfStrPred(semantic_embs, strings[:,1],weight=wt)['loss']
            SibStrPredLoss   = self.SibStrPred(semantic_embs, strings.gather(1,sib_str_index).squeeze(), weight=wt)['loss']
            # LeftStrPredLoss  = self.LeftStrPred(semantic_embs, strings[:,0],weight=wt)['loss']
            # RightStrPredLoss = self.RightStrPred(semantic_embs, strings[:,2],weight=wt)['loss']

            # Validity prediction task
            if self.num_aux_tasks == 8:
                sem_embs_with_neg, sem_labels_with_neg = self.generateNegSamples(semantic_embs)
                SemValPredLoss = self.SemValPred(sem_embs_with_neg, sem_labels_with_neg)
            
                syn_embs_with_neg, syn_labels_with_neg = self.generateNegSamples(syntactic_embs)
                SynValPredLoss = self.SynValPred(syn_embs_with_neg, syn_labels_with_neg)

                TaskLoss = torch.stack((SynPosPredLoss, SelfCatPredLoss, SibCatPredLoss, SemPosPredLoss, SelfStrPredLoss, SibStrPredLoss, SynValPredLoss, SemValPredLoss))

            else:
                TaskLoss = torch.stack((SynPosPredLoss, SelfCatPredLoss, SibCatPredLoss, SemPosPredLoss, SelfStrPredLoss, SibStrPredLoss))

            # TaskLoss = torch.stack((SynPosPredLoss, SynValPredLoss, SelfCatPredLoss, SibCatPredLoss, SemPosPredLoss, SemValPredLoss, SelfStrPredLoss, SibStrPredLoss))
            # TaskLoss = torch.stack((SynPosPredLoss, SelfCatPredLoss, SibCatPredLoss, SemPosPredLoss, SelfStrPredLoss, SibStrPredLoss))
            # TaskLoss = torch.stack((SynPosPredLoss, SelfCatPredLoss, SibCatPredLoss, SemPosPredLoss, SelfStrPredLoss, SibStrPredLoss, SynValPredLoss, SemValPredLoss))


            # TaskLoss = torch.stack((LeftCatPredLoss, SelfCatPredLoss, RightCatPredLoss, LeftStrPredLoss, SelfStrPredLoss, RightStrPredLoss))
            
        else:    TaskLoss = torch.zeros(self.num_aux_tasks, device=device)
        
        TaskLoss = TaskLoss * self.aux_mask.to(device)
        # self.weight = self.weight * self.aux_mask.to(device)

        # loss = torch.dot(TaskLoss,torch.exp(-self.weight))+ (sum(self.weight**2)**0.5)/2
        loss = torch.dot(TaskLoss, 1 / (1e-10 + 2 * self.weight**2)) + sum(torch.log(1 + (self.weight* self.aux_mask.to(device))**2))
        return {"loss": loss, "TaskLoss": TaskLoss, "syntactic_embs": syntactic_embs[root_index,:], "semantic_embs": semantic_embs[root_index,:]}
    
    ''' emb: FloatTensor(N,d) -> emb_with_neg: FloatTensor(N,neg+1,d), labels_with_neg: LongTensor(N,neg+1) '''
    def generateNegSamples(self, emb):
        N = emb.size(0)
        # noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(N,emb.size(1)),torch.stack([self.stdev]*N,0))
        # noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(N,emb.size(1)),torch.stack([torch.eye(emb.size(1))]*N,0))
        emb_with_neg, labels_with_neg = [emb],[[1]*N]    
        while len(emb_with_neg)<self.neg_samples+1:
            # noise = noise_sampler.sample().to(next(self.parameters()).device)  #(N,d)
            if emb.size(1)==self.dx:
                noise = self.syn_noise_sampler.rsample([emb.size(0)]).to(next(self.parameters()).device)  #(N,d)
            else:
                noise = self.sem_noise_sampler.rsample([emb.size(0)]).to(next(self.parameters()).device)  #(N,d)

            if torch.norm(noise)==0:    
                print('noise is too small')
                continue
            emb_with_neg.append(emb+noise)
            labels_with_neg.append([0]*N)

        emb_with_neg = torch.stack(emb_with_neg)
        labels_with_neg = torch.LongTensor(labels_with_neg).to(next(self.parameters()).device)
        
        return emb_with_neg, labels_with_neg
    

