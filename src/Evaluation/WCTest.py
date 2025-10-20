import argparse, time, datetime
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.linear_model import LinearRegression
# import numpy as np

cache_path = '../../data/cache/'

import time 

def evaluateWC(filename): 
    Data = pickle.load(open(filename,'rb'))
    TrX,TrY = Data['train']
    VaX,VaY = Data['val']
    TeX,TeY = Data['test']

    model = MLPClassifier(hidden_layer_sizes=(1000), early_stopping=True, random_state=1111, activation='identity')
    model.fit(TrX,TrY)
    acc = model.score(TeX,TeY)
    tracc = model.score(TrX,TrY)
    print('LinActMLP (1000)',filename.split('/')[-1][:-5],acc, tracc)

    model = MLPClassifier(hidden_layer_sizes=(500), early_stopping=True, random_state=1111, activation='identity')
    model.fit(TrX,TrY)
    acc = model.score(TeX,TeY)
    tracc = model.score(TrX,TrY)
    print('LinActMLP (500)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(early_stopping=True, random_state=1111)
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('MLP (100)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(hidden_layer_sizes=(50), early_stopping=True, random_state=1111)
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('MLP (50)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(hidden_layer_sizes=(), early_stopping=True, random_state=1111)
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('MLP (0)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(early_stopping=True, random_state=1111, activation='identity')
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('LinActMLP (100)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(hidden_layer_sizes=(50), early_stopping=True, random_state=1111, activation='identity')
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('LinActMLP (50)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, random_state=1111)
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('MLP (50,50)',filename.split('/')[-1][:-5],acc, tracc)

    # model = MLPClassifier(hidden_layer_sizes=(100,100), early_stopping=True, random_state=1111)
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('MLP (100,100)',filename.split('/')[-1][:-5],acc, tracc)

    # model = SVC()
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('SVM rbf',filename.split('/')[-1][:-5],acc, tracc)

    # model = SVC(kernel='linear')
    # model.fit(TrX,TrY)
    # acc = model.score(TeX,TeY)
    # tracc = model.score(TrX,TrY)
    # print('SVM linear',filename.split('/')[-1][:-5],acc, tracc)

    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
   
    t0=time.time()    
    # files = [filename for filename in os.listdir(cache_path) if filename.endswith('.pckl')]
    files = ['word_content-codi.pckl','word_content-codi_init_modelroberta.pckl','word_content-glove.pckl', 'word_content-roberta.pckl']
    for f in files: evaluateWC(cache_path+f)
    print('Total Execution Time:',datetime.timedelta(seconds=time.time()-t0))
    
  

