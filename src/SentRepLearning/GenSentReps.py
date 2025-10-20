
import numpy as np
import torch
import json, pickle, os, argparse
from tqdm import tqdm

from Loader import DataLoader 
from CoDi import CoDiConfig,CoDiModel

model_path = '../../models'
cache_path = '../../data/cache/'

class Representation:
    def __init__(self,dataset, loader, model):
        self.dataset = dataset
        self.model = model
        self.loader = loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model is not None:  self.model.to(self.device)
        
    def __getSentEmb(self,item):
        if self.model is None:
            embs = [emb for emb in item['sem_embs'] if torch.norm(emb)!=0]
            if len(embs)==0:    embs= [emb for emb in item['sem_embs']]
            x = torch.mean(torch.stack(embs),axis=0).cpu().numpy()
        else:
            with torch.no_grad():
                try:
                    item = {k:item[k].to(self.device) for k in item}
                    x = self.model(**(item), aux_tasks=False)  #(2), (batch,dm)
                except: 
                    print(item)
                    raise('Exception')
                x = torch.cat([x['syntactic_embs'],x['semantic_embs']],1).squeeze().cpu().numpy()

        return x

    def __getSICKERep(self, data):
        X = [(self.__getSentEmb(items[0]),self.__getSentEmb(items[1])) for items,label in data]
        X = [np.concatenate([np.abs(x1 - x2), x1 * x2]) for x1,x2 in X]
        Y = [label for items,label in data]
        return np.array(X), np.array(Y)

    def getRepresentation(self, parsed_data):
        data = [([self.loader.convertToTensor(**ss) for ss in s], l) for s,l in tqdm(parsed_data)]

        if self.dataset=='sicke':   return self.__getSICKERep(data)
        
        return np.array([self.__getSentEmb(items[0]) for items,label in tqdm(data)]), np.array([label for items,label in data])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='subj_number', help='Name of the dataset (sts/trec/sick/?)')
    parser.add_argument('--modelname', type=str, default='glove', help='Leaf emb initialization model glove/roberta')
    
    args = parser.parse_args()

    print(args.dataset,args.modelname)

    if args.modelname == 'glove' or args.modelname =='roberta' or args.modelname =='llama':    
        loader = DataLoader(max_str_len = 100, sem_emb_init_model = args.modelname)
        rep = Representation(args.dataset, loader, None)
    else:
        path_to_model = model_path+'/'+args.modelname
        assert os.path.isdir(path_to_model)
        ini_dim = json.load(open(path_to_model+'/'+'config.json'))['init_dim']
        if ini_dim == 300:  loader = DataLoader(max_str_len = 100, sem_emb_init_model = 'glove')
        elif ini_dim == 768:    loader = DataLoader(max_str_len = 100, sem_emb_init_model = 'roberta')
        elif ini_dim == 4096:    loader = DataLoader(max_str_len = 100, sem_emb_init_model = 'llama')
        else:   raise('Unknown init model')
        model = CoDiModel.from_pretrained(path_to_model, config = CoDiConfig.from_pretrained(path_to_model)) 
        rep = Representation(args.dataset, loader, model)


    # print('Modelname:',args.modelname)
    # if model is None:   print('None model')
    # else:   print('Trained codi model')

    parsed_data_filename = cache_path+args.dataset+'_parsed.json'

    if not os.path.exists(parsed_data_filename):
        raise("Please run preparse.py to parse data")
        
    parsed_data = json.load(open(parsed_data_filename))
    if 'test' not in parsed_data:   parsed_data['test'] = []
    if 'val' not in parsed_data:   parsed_data['val'] = []

    print('Parsed data loaded')
    
    Data = {}
    Data['train'] = rep.getRepresentation(parsed_data['train'])
    Data['val'] = rep.getRepresentation(parsed_data['val'])
    Data['test'] = rep.getRepresentation(parsed_data['test'])
    pickle.dump(Data,open(cache_path+args.dataset+'-'+args.modelname+'.pckl','wb'))
    print('Representations generated')

