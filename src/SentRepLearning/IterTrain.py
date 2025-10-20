import argparse, time, datetime

import json
import torch
from torch import nn
torch.manual_seed(0)

import os, math

from transformers import TrainingArguments, Trainer

from Loader import DataLoader
from CoDi import CoDiConfig,CoDiModel

cache_path = '../../data/cache/'

def collate_function(data):
    item = {}
    # print(data)
    item['adj_list'] = data[0]['adj_list'] 
    item['root_index'] = torch.LongTensor([0])
    for sample in data[1:]:
        adj = sample['adj_list']+ item['adj_list'].size()[0] 
        ri = sample['root_index']+ item['adj_list'].size()[0]

        item['adj_list'] = torch.cat((item['adj_list'],adj))                    # [(N,2)]*b -> (N*b,2)
        item['root_index'] = torch.cat((item['root_index'],ri))                 # [(1)*b] -> (b)
    
    item['node_order'] = torch.cat([sample['node_order'] for sample in data])   # [(N)]*b -> (N*b)
    item['sem_embs'] = torch.cat([sample['sem_embs'] for sample in data])       # [(N,dm)]*b -> (N*b,dm)
    item['syn_cats'] = torch.cat([sample['syn_cats'] for sample in data])       # [(N,3)]*b -> (N*b,3)
    item['strings'] = torch.cat([sample['strings'] for sample in data])         # [(N,3,100)]*b -> (N*b,3,100)
    # item['labels'] = torch.stack([sample['labels'] for sample in data]).squeeze()# [(1)]*b -> (b) or [(max_str_len)]*b -> (b,max_str_len)

    return item    

class DataSet(torch.utils.data.Dataset):
    def __init__(self,Data):
        self.Data = Data       
        
    def __getitem__(self,idx):
        item = {}
        item['adj_list'] = self.Data[idx]['adj_list']
        item['node_order'] = self.Data[idx]['node_order']
        item['sem_embs'] = self.Data[idx]['sem_embs']
        item['syn_cats'] = self.Data[idx]['syn_cats']
        item['strings'] = self.Data[idx]['strings']
        item['root_index'] = self.Data[idx]['root_index']
        # item['labels'] = self.Data[idx]['labels'] 
        
        return item
    
    def __len__(self):
        return len(self.Data)

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, size, loader):#, skipcache=False):
        super().__init__()
        self.epoch = 0
        self.loader = loader
        self.generator = torch.Generator()
        self.generator.manual_seed(2147483647)

        parsed_data_filename = cache_path+dataset+'_parsed.json'

        if not os.path.exists(parsed_data_filename): # or skipcache:
            raise("Please run preparse.py to parse data")
        
        self.parsed_data = json.load(open(parsed_data_filename))
        self.parsed_data['train'] = self.parsed_data['train'][:size]
        self.parsed_data['test'] = []
        self.size = len(self.parsed_data['train'])

        print(self.size, 'parsed samples loaded')
        print(sum([len(ss['strings'][0][1]) for s in self.parsed_data['train'] for ss in s[0]]), 'token')

    def __iter__(self):
        # print("\nThis is the epoch: ", self.epoch, "\n")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start = 0
            end = self.size
        else: # return the share of worker
            per_worker = int(math.ceil(self.size / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.size)
            # print(worker_info.num_workers, "number of workers")
        for idx in range(start,end):
            for s in self.parsed_data['train'][idx][0]:
                # print(idx)
                yield  self.loader.convertToTensor(**s)
               
    def __len__(self):
        return self.size 

'''Train the CoDi model (compositional operator)'''
def Train(model, loader, args):
        
    # TrainData, _  = loader(args.dataset, args.size, args.skipcache)  
    # dataset = DataSet([item for data,label in TrainData for item in data])

    dataset = IterDataset(dataset=args.dataset,size=args.size, loader=loader)#,skipcache=args.skipcache)
    
    training_args = TrainingArguments(
        # save_strategy               = 'epoch',
        save_strategy               = 'no',
        output_dir                  = '../../models/'+args.name,
        num_train_epochs            = args.epoch,
        per_device_train_batch_size = args.bs,
        weight_decay                = args.l2penalty,
        logging_dir                 = 'logs',
        logging_strategy            = 'epoch',
        learning_rate               = args.lr,
        report_to                   = "none",
        optim                       = "adamw_torch",
        disable_tqdm                = True,
        # lr_scheduler_type           = 'constant',
        save_only_model             = True,
        # save_total_limit            = 1,
    )
    
    trainer = Trainer(
        model           = model,                    
        args            = training_args,                  
        train_dataset   = dataset,         
        data_collator   = collate_function,
    )
        
    trainer.train()

    trainer.save_model('../../models/'+args.name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    '''Hyperparamters related to Dataset'''
    parser.add_argument('--compose', type=str, default='htlstm', help='name of the compositional operator: hypertransformer/transformer/treelstm/htlstm/hatlstm/linear/mean/sum/')

    parser.add_argument('--dataset', type=str, default='wiki', help='Name of the dataset (sts/trec/sick/wiki)')    
    parser.add_argument('--size', type=int, default=100000, help='Number of data samples used from each of the datasets') 

    parser.add_argument('--weighted_aux', action='store_true',help='Set for node-wise-weighted auxiliary task loss')
    parser.add_argument('--aux_mask', type=str, default='111111',help='Mask auxiliary tasks')

    parser.add_argument('--max_str_len', type=int, default=100, help='Maximum number of tokens generated for auxiliary tasks')
    parser.add_argument('--max_nodes_per_batch', type=int, default=128, help='Maximum number of nodes per batch considering for text generation auxiliary tasks')

    parser.add_argument('--skipcache', action='store_true',help='Skip cached data')
    
    '''Hyperparameters related to the Model'''
    parser.add_argument('--init_model', type=str, default='glove', help='Leaf emb initialization model glove/roberta')
    parser.add_argument('--neg_samples', type=int, default=10, help='Number of negative samples for semantic validity predicion task')
    # parser.add_argument('--C', type=int, default=100, help='Maximum number of syntactic categories. This will be the dimension of one-hot representation of leaf syntactic category')
    parser.add_argument('--dim', type=int, default=1024, help='Dimension of semantic embedding')
    parser.add_argument('--dx', type=int, default=64, help='Dimension of syntactic embedding')
    parser.add_argument('--tgt_h_dim', type=int, default=100, help='Hidden layers in target prediction model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in Transformer')

    '''Hyperparameters related to training'''
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs for finetuning')
    parser.add_argument('--bs', type=int, default=8, help='Batch size. 0 means a single batch')    
    parser.add_argument('--l2penalty', type=float, default=0.0, help='L2 Regularization penality weight')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    parser.add_argument('--toy', action='store_true',help='Test it on toy dataset and training epochs')

    parser.add_argument('--name', type=str, default='', help='Name of the dataset (trec/mr/sick/cogs)')

    default_dict = {k.option_strings[0][2:]: k.default for k in parser._actions}
    args = parser.parse_args()
    args.name = "_".join(["codi"]+[f"{key}{value}" for key, value in vars(args).items() if value != default_dict[key]])
    
    if args.toy:    
        args.size = 4
        args.epoch = 5
                 
    t0=time.time()
    
    loader = DataLoader(max_str_len = args.max_str_len, sem_emb_init_model=args.init_model)

    codiconfig = CoDiConfig(max_str_len = args.max_str_len, 
                            max_nodes_per_batch = args.max_nodes_per_batch, 
                            # init_dim = 300,
                            init_dim = loader.init_dim, 
                            # C = args.C, 
                            C = len(loader.cats),
                            dim = args.dim, 
                            dx  = args.dx, 
                            compose = args.compose, 
                            neg_samples = args.neg_samples, 
                            dropout = args.dropout, 
                            weighted_aux = args.weighted_aux,
                            aux_mask = args.aux_mask,
                        )
    codi = CoDiModel(codiconfig)
    codi.printParamCount()

    for p in codi.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        
    print('Args:',json.dumps(vars(args), indent=4))
    
    Train(model=codi, loader=loader, args=args)
    
    print('Total Execution Time:',datetime.timedelta(seconds=time.time()-t0))
