import numpy as np
from IsoScore import IsoScore

import torch
from torch import nn
import os, json, argparse

from CoDi import CoDiConfig, CoDiModel
from Loader import DataLoader
from GenSentReps import Representation
        
model_path = '../../models'
cache_path = '../../data/cache/'
# dx = 64
# noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(dx),torch.eye(dx))
class CoDiModel(CoDiModel):
    def __init__(self,config):
        super().__init__(config)  
                   
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
          
        return {"syntactic_embs": syntactic_embs, "semantic_embs": semantic_embs}
    
class Analysis:
    def __init__(self,modelname):
        print('Analysing',modelname)
        self.dataset = 'wiki'
        self.skip100K = True

        path_to_model = model_path+'/'+modelname
        assert os.path.isdir(path_to_model)
        ini_dim = json.load(open(path_to_model+'/'+'config.json'))['init_dim']
        dx = json.load(open(path_to_model+'/'+'config.json'))['dx']
        if ini_dim == 300:  self.init_model = 'glove'
        elif ini_dim == 768:  self.init_model = 'roberta'
        elif ini_dim == 4096:  self.init_model = 'llama'
        else:   raise('Unknown init model')
        self.loader = DataLoader(max_str_len = 100, sem_emb_init_model = self.init_model)
        self.model = CoDiModel.from_pretrained(path_to_model, config = CoDiConfig.from_pretrained(path_to_model)) 

        self.noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(dx),torch.eye(dx))

    def loadData(self, size=10000):
        print('Loading',self.dataset)

        parsed_data = json.load(open(cache_path+self.dataset+'_parsed.json'))['train']

        if self.skip100K:    parsed_data = parsed_data[100000:] 

        data = [([self.loader.convertToTensor(**ss) for ss in s], l) for s,l in parsed_data[:size]]
        print('Number of samples:',len(data))

        return data

    def getNorm(self,e):
        e = np.array(e)
        norm = np.linalg.norm(e,axis=1)
        fq = np.percentile(norm,25)
        med = np.percentile(norm,50)
        lq = np.percentile(norm,75)
        mn = min(norm)
        mx = max(norm)
        mean = np.mean(norm)

        return mn, fq, med, lq, mx, mean, e.shape[0]

    def getScore(self,e):
        e = np.array(e)
        norm = np.linalg.norm(e,axis=1)
        iso = 0.0 if e.shape[0]<=1 else round(float(IsoScore.IsoScore(e)),4)
        return min(norm), max(norm), np.mean(norm), iso, e.shape[0]

    def printCatFreq(self):
        parsed_data = json.load(open(cache_path+'wiki_parsed.json'))['train'][:100000]
        cat_freq = {}
        for items,l in parsed_data:
            for node in items[0]['syn_cats']:
                cat = node[1]
                if node[1] not in cat_freq: cat_freq[node[1]]=0
                cat_freq[node[1]]+=1
        for cat in cat_freq:
            print(cat,cat_freq[cat])

    def baseline(self):
        rep = Representation(self.dataset, self.loader, None)
        parsed_data = json.load(open(cache_path+self.dataset+'_parsed.json'))['train']
        if self.skip100K:    parsed_data = parsed_data[100000:] 
        embs = rep.getRepresentation(parsed_data[:10000])[0]
        return self.getNorm(embs)
        # print('AvgEmb, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {}'.format(*self.getNorm(embs)))

    def baseScore(self,size):
        rep = Representation(self.dataset, self.loader, None)
        parsed_data = json.load(open(cache_path+self.dataset+'_parsed.json'))['train']
        if self.skip100K:    parsed_data = parsed_data[100000:] 
        embs = rep.getRepresentation(parsed_data[:size])[0]
        return self.getScore(embs)

    def printHeightWiseScores(self, maxsize=20000, size=1088):
        print('Task: Print Height-wise scores')
        data = self.loadData(maxsize)
        root = []
        embs = {}
        for items,_ in data:
            with torch.no_grad():   x = self.model(**(items[0]), aux_tasks=False)                
            root.append(np.concatenate([x['syntactic_embs'][0,:].squeeze().cpu().numpy(),x['semantic_embs'][0,:].squeeze().cpu().numpy()]))

            node_order = [int(n) for n in items[0]['node_order']]

            for i,n in enumerate(node_order):
                if n not in embs:  embs[n] = []
                embs[n].append(np.concatenate([x['syntactic_embs'][i,:].squeeze().cpu().numpy(),x['semantic_embs'][i,:].squeeze().cpu().numpy()]))
        
        print('Height, Min-Norm, Max-Norm, Mean-Norm, IsoScore, Size')
        print('AvgEmb, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(*self.baseScore(size)))
        print('Root, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(*self.getScore(root)))

        for i in sorted(embs.keys()):
            if len(embs[i])<size:  continue
            print('{:0=2d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(i, *self.getScore(embs[i][:size])))
            
    def printHeightWiseNorm(self, size=10000):
        print('Task: Print Height-wise norm')
        data = self.loadData(size)
        SynRoot,SemRoot,CombRoot = [],[],[]
        Syn, Sem, Comb = {}, {}, {}
        for items,_ in data:
            with torch.no_grad():   x = self.model(**(items[0]), aux_tasks=False)
            
            SynRoot.append(x['syntactic_embs'][0,:].squeeze().cpu().numpy())
            SemRoot.append(x['semantic_embs'][0,:].squeeze().cpu().numpy())
            CombRoot.append(np.concatenate([x['syntactic_embs'][0,:].squeeze().cpu().numpy(),x['semantic_embs'][0,:].squeeze().cpu().numpy()]))

            node_order = [int(n) for n in items[0]['node_order']]

            for i,n in enumerate(node_order):
                if n not in Syn:  Syn[n] = []
                if n not in Sem:  Sem[n] = []
                if n not in Comb:  Comb[n] = []

                Syn[n].append(x['syntactic_embs'][i,:].squeeze().cpu().numpy())
                Sem[n].append(x['semantic_embs'][i,:].squeeze().cpu().numpy())
                Comb[n].append(np.concatenate([x['syntactic_embs'][i,:].squeeze().cpu().numpy(),x['semantic_embs'][i,:].squeeze().cpu().numpy()]))
        
        print('Shape of combined representations',np.array(CombRoot).shape)
        print('Height, Min, p25, p50, p75, Max, Mean, Size')
        print('AvgEmb, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(*self.baseline()))
        print('Root, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(*self.getNorm(CombRoot)))

        for i in sorted(Syn.keys()):
            if len(Syn[i])==0:  continue
            print('{:0=2d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {} '.format(i, *self.getNorm(Comb[i])))
        
    def analyseCompOp(self, size=100,num_samples=10000):
        print('Task: Analyse Compop')
        cat_scores = {}
        node_sem_embs = []
        data = self.loadData(size)
        for items,_ in data:
            with torch.no_grad():   x = self.model(**(items[0]), aux_tasks=False)
            for i, a in enumerate(items[0]['adj_list']):
                if int(a[0])==i and int(a[1])==i:   continue # leaf node
                node_sem_embs.append(x['semantic_embs'][i,:].detach().numpy())
                el = x['semantic_embs'][int(a[0]),:].repeat(50,1)
                er = x['semantic_embs'][int(a[1]),:].repeat(50,1)
                hn = x['syntactic_embs'][i,:]
                embs = torch.concatenate([self.model.semantic_compose(el, er, None, hn+self.noise_sampler.rsample([50])) for _ in range(int(num_samples/50))])
                embs = np.array(embs.detach())
                norm = self.getNorm(embs-x['semantic_embs'][i,:].detach().numpy())
                assert norm[-1]==num_samples and embs.shape[0]==num_samples
                cat = int(items[0]['syn_cats'][i][1])
                if cat not in cat_scores:   cat_scores[cat] = [[],[]]
                cat_scores[cat][0].append(float(IsoScore.IsoScore(embs)))
                cat_scores[cat][1].append(norm[5])

        node_sem_embs = np.array(node_sem_embs)
        node_iso = float(IsoScore.IsoScore(node_sem_embs))
        centroid = np.mean(node_sem_embs, axis=0)
        assert centroid.shape[0]==node_sem_embs.shape[1]
        node_spread = self.getNorm(node_sem_embs-centroid)
        print('Node-baseline:',round(node_iso,4),round(node_spread[5],4),node_sem_embs.shape[0])
        for cat in cat_scores:
            print(self.loader.cats[cat],round(sum(cat_scores[cat][0])/len(cat_scores[cat][0]),4), round(sum(cat_scores[cat][1])/len(cat_scores[cat][1]),4))
        tot0 = [s for cat in cat_scores for s in cat_scores[cat][0]]
        tot1 = [s for cat in cat_scores for s in cat_scores[cat][1]]
        print('Average:',round(sum(tot0)/len(tot0),4),round(sum(tot1)/len(tot1),4))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--printnorm', action='store_true',help='Analyse embedding space and print height wise norm')
    parser.add_argument('--printscore', action='store_true',help='Analyse embedding space and print height wise isoscore and norm')
    parser.add_argument('--companalyse', action='store_true',help='Analyse effect of hypernetwork on composition')
    parser.add_argument('--catfreq', action='store_true',help='Print frequency of categories in wiki train data')
    parser.add_argument('--model', type=str, default='codi_init_modelroberta', help='')

    args = parser.parse_args()

    A = Analysis(args.model)
    if args.printnorm:      A.printHeightWiseNorm(size=10000)
    if args.printscore:     A.printHeightWiseScores(maxsize=20000, size=1088)
    if args.companalyse:    A.analyseCompOp(100,10000)
    if args.catfreq:        A.printCatFreq()
