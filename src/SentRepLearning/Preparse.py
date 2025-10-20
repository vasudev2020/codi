
from Loader import DataLoader, Parser
import json, argparse
from tqdm import tqdm

# dataset = 'wiki'
cache_path = '../../data/cache/'

def parseData(dataset, size=None):
    loader = DataLoader(max_str_len = 100)
    parser = Parser()
    parsed_data_filename = cache_path+dataset+'_parsed.json'

    parsed_data = {'train':[]}

    trdata, tsdata, vadata = loader.loadData(dataset,size) if size is not None else loader.loadData(dataset)

    with open(parsed_data_filename, 'w') as fp: 
        for s,l in tqdm(trdata):
            try:    parsed_data['train'].append(([parser(sent) for sent in s],l))
            except: print("Parsing error:",' '.join(s)) 

        if len(tsdata)!=0:  parsed_data['test']=[]
        for s,l in tqdm(tsdata):
            try:    parsed_data['test'].append(([parser(sent) for sent in s],l))
            except: print("Parsing error:",' '.join(s))  

        if len(vadata)!=0:  parsed_data['val']=[]
        for s,l in tqdm(vadata):
            try:    parsed_data['val'].append(([parser(sent) for sent in s],l))
            except: print("Parsing error:",' '.join(s))

        json.dump(parsed_data, fp)


                           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='subj_number', help='name of dataset')
    parser.add_argument('--size', type=int, default=None, help='number of samples to be parsed')

    args = parser.parse_args()
    parseData(args.dataset, args.size)