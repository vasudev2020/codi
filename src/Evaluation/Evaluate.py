import argparse, time, datetime, os
import pandas as pd
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

cache_path = '../../data/cache/'

def evaluate(args): 
    method, filename, dataset = args
    try:
        Data = pickle.load(open(filename,'rb'))
    except: 
        print("Pickle data cannot be opened")
        return 0.0
    TrX,TrY = Data['train']
    VaX,VaY = Data['val']
    TeX,TeY = Data['test']

    hidd_layer = () if dataset=='word_content' else (50)
    # model = MLPClassifier(hidden_layer_sizes=(), early_stopping=True, random_state=1111) if dataset=='word_content' else MLPClassifier(early_stopping=True, random_state=1111)
    model = MLPClassifier(hidden_layer_sizes=hidd_layer, early_stopping=True, random_state=1111)

    # print(TrX.shape,TrY.shape)
    if len(TeX)>0:
        model.fit(TrX,TrY)
        acc = model.score(TeX,TeY)
        # acc_val = model.score(*Data['val'])
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        acc = []
        for train_idx, test_idx in skf.split(TrX, TrY):
            model.fit(TrX[train_idx],TrY[train_idx])
            acc.append(model.score(TrX[test_idx],TrY[test_idx]))
        acc = sum(acc)/len(acc)

    print(filename.split('/')[-1][:-5],acc)

    return round(acc,4)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--classifier', type=str, default='sklearn', help='sklearn/')
    parser.add_argument('--benchmark', type=str, default='probing', help='probing/')

    args = parser.parse_args()
                     
    t0=time.time()
    if args.benchmark=='probing':
        datasets = ['subj_number', 'obj_number', 'bigram_shift', 'odd_man_out', 'coordination_inversion', 'top_constituents', 'tree_depth', 'past_present', 'sentence_length', 'word_content']
    
    files = [filename for filename in os.listdir(cache_path) if filename.endswith('.pckl')]
    inputs = [[args.classifier, cache_path+f, f.split('-')[0]] for f in files if f.split('-')[0] in datasets]

    outputs = [evaluate(i) for i in inputs]
    results = {}
    for i,o in zip(inputs,outputs):
        _, filename, dataset = i
        if dataset not in results:  results[dataset] = {}
        results[dataset][filename.split('-')[-1][:-5]] = o

    res = pd.DataFrame(results)
    res.to_csv(open(f'scores_{args.classifier}_{args.benchmark}.csv','w'))

    print('Total Execution Time:',datetime.timedelta(seconds=time.time()-t0))
    