from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import ModernBertModel

import os, torch
from sklearn.neural_network import MLPClassifier

data_path = '../../data/'

def textClean(text):
    text = text.replace('\\n','').replace('\\"','"').replace('\\','').replace(':','').replace('.','').strip()       
    text = text.encode('ascii',errors='ignore').decode()
    text = ' '.join(text.split())
    return text

def readProbingData(task):
    datafilename = data_path+'ProbingTasks/'+task+'.txt'
    if not os.path.exists(datafilename):
        raise Exception(f"Invalid dataset: {task}")

    trdata,tedata,vadata =[],[],[]
    data = open(datafilename).readlines()
    L = list(set([line.split('\t')[1] for line in data]))
    for line in data:
        fields = line.split('\t')
        assert len(fields)==3
        if fields[0]=='tr': trdata.append(([textClean(fields[2])], L.index(fields[1])))
        if fields[0]=='te': tedata.append(([textClean(fields[2])], L.index(fields[1])))
        if fields[0]=='va': vadata.append(([textClean(fields[2])], L.index(fields[1])))

    return trdata,tedata,vadata

class Baseline:
    def __init__(self, modeltype, modelname):
        self.modeltype = modeltype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if modeltype=='sbert':
            self.model = SentenceTransformer(modelname)
        elif modeltype=='SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained(modelname)
            self.model = AutoModel.from_pretrained(modelname).to(self.device)
        elif modeltype=='mbert':
            self.tokenizer = AutoTokenizer.from_pretrained(modelname)
            self.model = ModernBertModel.from_pretrained(modelname).to(self.device)

    def Encode(self,texts):
        if self.modeltype=='sbert': return self.model.encode(texts)
        if self.modeltype=='SimCSE':
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings.append(self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.squeeze().cpu().tolist())
            return embeddings
        if self.modeltype=='mbert':
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings.append(self.model(**inputs).last_hidden_state[0].mean(axis=0).cpu().tolist())
                assert len(embeddings[0])==768
            return embeddings

    def Evaluate(self):
        # datasets = ['subj_number', 'obj_number', 'bigram_shift', 'odd_man_out', 'coordination_inversion', 'top_constituents', 'tree_depth', 'past_present', 'sentence_length', 'word_content']
        datasets = ['sentence_length', 'tree_depth', 'top_constituents', 'past_present', 'subj_number', 'obj_number', 'coordination_inversion', 'word_content']
        for dataset in datasets:
            trdata,tedata,vadata = readProbingData(dataset)
            TrX = self.Encode([t[0] for t,l in trdata])
            TeX = self.Encode([t[0] for t,l in tedata])
            TrY = [l for t,l in trdata]
            TeY = [l for t,l in tedata]

            hidd_layer = () if dataset=='word_content' else (50)
            mlp = MLPClassifier(hidden_layer_sizes=hidd_layer, early_stopping=True, random_state=1111)
            mlp.fit(TrX,TrY)
            acc = mlp.score(TeX,TeY)
            print(dataset,acc)

# model_type = "SimCSE"
# model_name = "princeton-nlp/sup-simcse-roberta-large"
# # model_name = "princeton-nlp/sup-simcse-roberta-base"

# model_type = 'mbert'
# model_name = 'answerdotai/ModernBERT-base'

model_type = 'sbert'
model_name = 'sentence-transformers/all-mpnet-base-v2'

print(f"{model_type}: {model_name}")
B = Baseline(modeltype=model_type, modelname=model_name)

B.Evaluate()


