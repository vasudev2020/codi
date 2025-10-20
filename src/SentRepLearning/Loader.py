
import os, json
import benepar
import spacy
from nltk.tokenize import sent_tokenize
import torch
import pandas as pd

from transformers import AutoTokenizer, RobertaModel, AutoModelForCausalLM

data_path = '../../data/'
cache_path = data_path+'cache/'
glove_file = '../../../Data/glove.840B.300d.txt'

hf_token = ''

class Parser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
            
    '''
    Input: A list of benepar parse tree objects
    Output: A binary tree [Node(Category), left_subtree, right_subtree] | [Category(str), Word(str)]
    '''
    def toBinaryTree(self,tree):
        if type(tree)==list:            
            if len(tree)==1:    return self.toBinaryTree(tree[0])
            left = self.toBinaryTree(tree[0])
            right = self.toBinaryTree(tree[1:])
            return ['Unk',left,right]
            # return [left[0]+'-'+right[0],left,right]
        
        children = list(tree._.children)
        if len(children)==0:  
            x = tree._.parse_string.replace('(','').replace(')','').split()
            if len(x)<=1:   print('some issue with parsing',x,tree._.parse_string)
            assert len(x)>1
            return [x[0],x[-1]]
         
        t = self.toBinaryTree(children)
        assert len(t)==3
        t[0] = tree._.labels[0]
        return t
    
    def formatTree(self,tree,baseindex):
        if len(tree)==2:    return [None], [['NULL', tree[0], 'NULL']], [tree[1]], [[[''],[tree[1]],['']]]
        left_adj_list, left_syn_cats, left_tokens, left_str = self.formatTree(tree[1], baseindex+1)
        right_adj_list, right_syn_cats, right_tokens, right_str  = self.formatTree(tree[2], baseindex+1 + len(left_adj_list))
        adj_list = [(baseindex+1, baseindex+1 + len(left_adj_list))] + left_adj_list + right_adj_list
        #syn_cats = [tree[0]]+left_syn_cats+right_syn_cats

        left_syn_cats[0][2] = right_syn_cats[0][1]
        right_syn_cats[0][0] = left_syn_cats[0][1]
        syn_cats = [['NULL', tree[0], 'NULL']] + left_syn_cats + right_syn_cats
        tokens = [None] + left_tokens + right_tokens
        
        # left_str = [t for t in left_tokens if t is not None]
        # right_str = [t for t in right_tokens if t is not None]
        left_str[0][2] = right_str[0][1]
        right_str[0][0] = left_str[0][1]
        strings = [[[''], left_str[0][1] + right_str[0][1], ['']]] + left_str + right_str
        
        return adj_list, syn_cats, tokens, strings
        
    def jointTrees(self, sentences):
        if len(sentences)==1:   return sentences[0]
        return ['SS', sentences[0], self.jointTrees(sentences[1:])]
    
    def __call__(self,sent,baseindex=0):
        # print(sent)        
        sentences = sent if type(sent) is list else [sent]
        sentences = [sent.strip() for sent in sentences if len(sent.strip())!=0]
        sentences = [' '.join(sent.split()) for sent in sentences]
        if len(sentences)==0:   raise Exception('Empty sentence to parse')
        trees = [self.toBinaryTree(list(self.nlp(sent).sents)) for sent in sentences]
        tree = self.jointTrees(trees)
        # print(tree)
        adj_list, syn_cats, tokens,strings = self.formatTree(tree,baseindex)

        node_order = [0 if a is None else -1 for a in adj_list]
        order = 1
        while True:
            if -1 not in node_order: break
        
            for i in range(len(node_order)):
                if node_order[i]==-1 and node_order[adj_list[i][0]] in range(order) and node_order[adj_list[i][1]] in range(order):
                    node_order[i] = order
            order += 1
        return {'adj_list': adj_list, 'syn_cats': syn_cats, 'tokens': tokens, 'strings': strings, 'node_order': node_order}

class DataLoader:
    def __init__(self, max_str_len=100, sem_emb_init_model='glove'):
        self.cats = ['ADJP','ADVP','CC','CD','CONJP','DT','EX','FRAG','FW','IN','INTJ','JJ','JJR','JJS','LS','LST','MD','NAC','NN','NNP','NNPS','NNS','NP','NX','PDT','POS','PP','PRN','PRP','PRP$','PRT','QP','RB','RBR','RBS','RP','RRC','S','SBAR','SBARQ','SINV','SQ','SYM','TO','UCP','UH','VB','VBD','VBG','VBN','VBP','VBZ','VP','WDT','WHADJP','WHADVP','WHNP','WHPP','WP','WP$','WRB','X','Unk','NULL','NFP','NML','HYPH','ADJ','-LRB-','-RRB-',':','.','``','$',"''",',','-ADV','-BNF','-CLF','-CLR','-DIR','-DTV','-EXT','-HLN','-LGS','-LOC','-MNR','-NOM','-PRD','-PRP','-PUT','-SBJ','-TMP','-TPC','-TTL','-VOC']

        self.max_str_len = max_str_len
        # self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

        self.SemEmbInitModel=sem_emb_init_model
        self.init_dim = 0
        if self.SemEmbInitModel=='glove':   self.init_dim = 300
        if self.SemEmbInitModel=='roberta': self.init_dim = 768    
        if self.SemEmbInitModel=='llama': self.init_dim = 4096        
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.NoneEmb = torch.zeros(self.init_dim).to(self.device)


    def textClean(self,text):
        text = text.replace('\\n','').replace('\\"','"').replace('\\','').replace(':','').replace('.','').strip()       
        text = text.encode('ascii',errors='ignore').decode()
        text = ' '.join(text.split())
        return text

    def readMR(self):

        data = [([self.textClean(line)],1) for line in open(data_path+'MR/rt-polarity.pos', 'r', encoding='latin-1').readlines()]
        data += [([self.textClean(line)],0) for line in open(data_path+'MR/rt-polarity.neg', 'r', encoding='latin-1').readlines()]

        # with open(data_path+'MR/rt-polarity.pos', 'r', encoding='latin-1') as f:
        #     data = [([self.textClean(line)],1) for line in f.read().splitlines()]

        # with open(data_path+'MR/rt-polarity.neg', 'r', encoding='latin-1') as f:
        #     data += [([self.textClean(line)],0) for line in f.read().splitlines()]

        return data,[],[]

    def readCR(self):
        labels = ['positive','negative']
        ds = json.load(open(data_path+'CR/CR_Train.json'))
        trdata = [([d["text"]],labels.index(d["label"])) for d in ds]

        ds = json.load(open(data_path+'CR/CR_Test.json'))
        tedata = [([d["text"]],labels.index(d["label"])) for d in ds]
        
        return trdata,tedata,[]

    def readSUBJ(self):
        data = [([self.textClean(line)],1) for line in open(data_path+'SUBJ/plot.tok.gt9.5000', 'r', encoding='latin-1').readlines()]
        data += [([self.textClean(line)],0) for line in open(data_path+'SUBJ/quote.tok.gt9.5000', 'r', encoding='latin-1').readlines()]

        return data,[],[]
        
    def readMPQA(self):

        df = pd.read_parquet(data_path+'MPQA/train-00000-of-00001-a7df005a1b07881a.parquet')
        trdata = [([s],l)for s,l in zip(df['sentence'],df['label'])]

        df = pd.read_parquet(data_path+'MPQA/dev-00000-of-00001-8814a3252cc44468.parquet')
        vadata = [([s],l)for s,l in zip(df['sentence'],df['label'])]

        df = pd.read_parquet(data_path+'MPQA/test-00000-of-00001-05fc5ca1c399669e.parquet')
        tedata = [([s],l)for s,l in zip(df['sentence'],df['label'])]

        return trdata,tedata,vadata

    def readSST2(self):
        df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/data/train-00000-of-00001.parquet")
        trdata = [([s],l)for s,l in zip(df['sentence'],df['label'])]
        
        df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/data/validation-00000-of-00001.parquet")
        vadata = [([s],l)for s,l in zip(df['sentence'],df['label'])]

        df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/data/test-00000-of-00001.parquet")
        tedata = [([s],l)for s,l in zip(df['sentence'],df['label'])]
        
        return trdata,tedata,vadata


    def readTREC(self):
        #lines = open('../Data/CoDiDataset/TREC/train_5500.label.txt',encoding = "ISO-8859-1").readlines()
        lines = open(data_path+'TREC/train.txt',encoding = "ISO-8859-1").readlines()
        trlabels = [l.split()[0].split(':')[0] for l in lines]
        trdata = [[self.textClean(' '.join(l.split()[1:]))] for l in lines]
        
        lines = open(data_path+'TREC/TREC_10.label.txt').readlines()
        tslabels = [l.split()[0].split(':')[0] for l in lines]
        tsdata = [[self.textClean(' '.join(l.split()[1:]))] for l in lines]
        
        labelindex = []
        for l in trlabels+tslabels:  
            if l not in labelindex: labelindex.append(l)
            
        trlabels = [labelindex.index(l) for l in trlabels]
        tslabels = [labelindex.index(l) for l in tslabels]

        trdata = [(s,l) for s,l in zip(trdata,trlabels)]
        tsdata = [(s,l) for s,l in zip(tsdata,tslabels)]
        
        return trdata,tsdata

    def readMRPC(self):
        lines = [l.strip().split('\t') for l in open(data_path+'MRPC/msr_paraphrase_train.txt').readlines()[1:]]
        trdata = [([self.textClean(l[3]),self.textClean(l[4])],int(l[0])) for l in lines]

        lines = [l.strip().split('\t') for l in open(data_path+'MRPC/msr_paraphrase_test.txt').readlines()[1:]]
        tedata = [([self.textClean(l[3]),self.textClean(l[4])],int(l[0])) for l in lines]

        return trdata,tedata

    def readSICKE(self):
        lines = open(data_path+'SICK/SICK.txt').readlines()[1:]
        trdata,trlabels,tsdata,tslabels,vadata,valabels = [],[],[],[],[],[]
        for l in lines:
            fields = [e.strip() for e in l.split('\t')]
            if fields[-1]=='TRAIN':
                trdata.append (fields[1:3])
                trlabels.append(fields[3]) #TODO: change field[3] to field[?] for score
            elif fields[-1]=='TEST':
                tsdata.append (fields[1:3])
                tslabels.append(fields[3]) #TODO: change field[3] to field[?] for score
            elif fields[-1]=='TRIAL':
                vadata.append (fields[1:3])
                valabels.append(fields[3]) 
            else:
                print('Unparsable fields:',fields)
                                
        labelindex = []
        for l in trlabels+tslabels+valabels:  
            if l not in labelindex: labelindex.append(l)
            
        trlabels = [labelindex.index(l) for l in trlabels]
        tslabels = [labelindex.index(l) for l in tslabels]
        valabels = [labelindex.index(l) for l in valabels]
        
        trdata = [(s,l) for s,l in zip(trdata,trlabels)]
        tsdata = [(s,l) for s,l in zip(tsdata,tslabels)]
        vadata = [(s,l) for s,l in zip(vadata,valabels)]

        return trdata, tsdata, vadata

    def readSST5(self):
        pass
    
    def readSTSB(self):
        trdata,tsdata,vadata = [],[],[]
        with open(data_path+"STS/sts-train.tsv", 'r') as fp:
            for line in fp.readlines():
                genre, filename, year, ids, score, sentence1, sentence2 = line.strip().split('\t')[:7]
                trdata.append(([sentence1, sentence2],float(score)))
                
        with open(data_path+"STS/sts-test.tsv", 'r') as fp:
            for line in fp.readlines():
                genre, filename, year, ids, score, sentence1, sentence2 = line.strip().split('\t')[:7]
                tsdata.append(([sentence1, sentence2],float(score)))

        with open(data_path+"STS/sts-dev.tsv", 'r') as fp:
            for line in fp.readlines():
                genre, filename, year, ids, score, sentence1, sentence2 = line.strip().split('\t')[:7]
                vadata.append(([sentence1, sentence2],float(score)))

        return trdata, tsdata, vadata

    def readProbingData(self, task):
        datafilename = data_path+'ProbingTasks/'+task+'.txt'
        if not os.path.exists(datafilename):
            raise Exception(f"Invalid dataset: {task}")

        trdata,tedata,vadata =[],[],[]
        data = open(datafilename).readlines()
        L = list(set([line.split('\t')[1] for line in data]))
        for line in data:
            fields = line.split('\t')
            assert len(fields)==3
            if fields[0]=='tr': trdata.append(([self.textClean(fields[2])], L.index(fields[1])))
            if fields[0]=='te': tedata.append(([self.textClean(fields[2])], L.index(fields[1])))
            if fields[0]=='va': vadata.append(([self.textClean(fields[2])], L.index(fields[1])))

        return trdata,tedata,vadata
    
    def readWiki(self,size):
        wiki_path='../../data/Wiki'
        # print(os.listdir('../../data/Wiki'))
        dirs = os.listdir(wiki_path)
        dataset = []
        for dir in dirs:
            files = os.listdir(wiki_path+'/'+dir)
            for f in files:
                for line in open(wiki_path+'/'+dir+'/'+f):
                    d = json.loads(line)
                    samples = d['text'].split('\n')
                    samples = [ss.strip() for s in samples for ss in sent_tokenize(s) if len(ss.strip())>0 and len(ss.split())<512]
                    dataset.extend(samples)
                    if len(dataset)>=size:
                        dataset = list(set(dataset))
                        if len(dataset)>=size:
                            return [([sent],0) for sent in dataset[:size]], []
        
        dataset = list(set(dataset))
        return [([sent],0) for sent in dataset]

    
    def readSample(self):
        return [(['This is a sample sentence'],0), (['How are you'],1)], [(['This is not a sample sentence'],0), (['How are you'],1)]

    def loadData(self, dataset='sample',size=1000000):
        tsdata, vadata = [],[]
        if dataset=='sample':   trdata,tsdata = self.readSample()
        
        elif dataset=='mr': trdata,tsdata,vadata = self.readMR()
        elif dataset=='cr': trdata,tsdata,vadata = self.readCR()
        elif dataset=='subj': trdata,tsdata,vadata = self.readSUBJ()
        elif dataset=='mpqa': trdata,tsdata,vadata = self.readMPQA()
        elif dataset=='sst2': trdata,tsdata,vadata = self.readSST2()
        elif dataset=='trec': trdata,tsdata = self.readTREC()
        elif dataset=='mrpc': trdata,tsdata = self.readMRPC()
        elif dataset=='sicke':   trdata, tsdata, vadata = self.readSICKE()

        elif dataset=='sst5': trdata,tsdata,vadata = self.readSST5()
        elif dataset=='sts':   trdata,tsdata = self.readSTSB()
        
        elif dataset=='wiki':   trdata = self.readWiki(size)
        
        else:  trdata,tsdata,vadata = self.readProbingData(dataset)
        
        return trdata[:size], tsdata[:size], vadata[:size]
    
    def initSemEmbs(self, terms):
        if self.SemEmbInitModel=='glove':
            if not hasattr(self,"gloveModel"):  
                f = open(glove_file,'r')
                self.gloveModel = {}
                for line in f:
                    splitLines = line.split()
                    if len(splitLines)!=301:    continue
                    self.gloveModel[splitLines[0]] = torch.FloatTensor([float(value) for value in splitLines[1:]])
                print(len(self.gloveModel)," words loaded!")
            return torch.stack([self.gloveModel[t] if t in self.gloveModel else self.NoneEmb.cpu() for t in terms])
        if self.SemEmbInitModel=='roberta':    # Initialize leaf terms with RoBERTa embeddings        
            if not hasattr(self,'roberta'):   self.roberta = RobertaModel.from_pretrained("roberta-base").to(self.device)

            filtered_terms = [t for t in terms if t is not None]

            pos = 0
            term_offset = []
            for t in filtered_terms:
                term_offset.append((pos,pos+len(t)))
                pos+=len(t)+1
            inputs = self.tokenizer(' '.join(filtered_terms), return_tensors="pt",truncation=True,return_offsets_mapping=True).to(self.device)
            out = self.roberta(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state[0][1:-1].detach()
            token_offsets = inputs['offset_mapping'][0][1:-1]
            assert out.size()[0]==len(token_offsets)

            #:This is a serious mistake. embs share memory
            # i = 0
            # embs = [[]]*len(filtered_terms)
            # for emb,tok_offset in zip(out,token_offsets):
            #     if tok_offset[1]>term_offset[i][1]:    i+=1
            #     assert tok_offset[0]>=term_offset[i][0] and tok_offset[1]<=term_offset[i][1]
            #     embs[i].append(emb)
            # print(token_offsets,term_offset)
            # raise Exception('Quit')
            embs = [[]]
            for emb,tok_offset in zip(out,token_offsets):
                if tok_offset[1]>term_offset[len(embs)-1][1]: embs.append([])
                assert tok_offset[0]>=term_offset[len(embs)-1][0] and tok_offset[1]<=term_offset[len(embs)-1][1]
                embs[-1].append(emb)

            embs = [torch.mean(torch.stack(e),0) for e in embs]

            try:
                stacked_embs = torch.stack([embs.pop(0) if t is not None else self.NoneEmb for t in terms])
                assert len(embs)==0
            except:
                print("Initialization failed:",' '.join(filtered_terms))
                stacked_embs = torch.stack([self.NoneEmb for t in terms])
            return stacked_embs.cpu()
        if self.SemEmbInitModel=='llama':    # Initialize leaf terms with RoBERTa embeddings  
            if not hasattr(self,'llama'):   self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", token=hf_token).to(self.device)
            if not hasattr(self,'llama_tokenizer'):   self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=hf_token)

            filtered_terms = [t for t in terms if t is not None]

            pos = 0
            term_offset = []
            for t in filtered_terms:
                term_offset.append((pos,pos+len(t)))
                pos+=len(t)+1

            inputs = self.llama_tokenizer(' '.join(filtered_terms), return_tensors="pt",truncation=True,return_offsets_mapping=True).to(self.device)
            # output = llama(**input_ids, output_hidden_states=True)
            output = self.llama(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], output_hidden_states=True)
            out = output.hidden_states[-1][0][1:].detach()
            token_offsets = inputs['offset_mapping'][0][1:]
            assert out.size()[0]==len(token_offsets)

            # print(token_offsets,term_offset)
            embs = [[]]
            for emb,tok_offset in zip(out,token_offsets):
                if tok_offset[1]>term_offset[len(embs)-1][1]: 
                    embs.append([])
                    # print(tok_offset, term_offset[len(embs)-1])
                assert tok_offset[0]+1>=term_offset[len(embs)-1][0] and tok_offset[1]<=term_offset[len(embs)-1][1]
                embs[-1].append(emb)

            embs = [torch.mean(torch.stack(e),0) for e in embs]

            stacked_embs = torch.stack([embs.pop(0) if t is not None else self.NoneEmb for t in terms])
            assert len(embs)==0
            return stacked_embs.cpu()       

    def toVocabIndex(self, seq):
        index = self.tokenizer(' '.join(seq), padding='max_length', truncation=True, max_length=self.max_str_len)["input_ids"]
        return index
    
    def detoknize(self,tokens):
        return self.tokenizer.decode(tokens)
            
    def convertToTensor(self, adj_list,  node_order, syn_cats, strings, tokens):    
        adj_list = torch.LongTensor([(i,i) if a is None else a for i,a in enumerate(adj_list)])
        node_order = torch.LongTensor(node_order)

        sem_embs = self.initSemEmbs(tokens)
        syn_cats = torch.LongTensor([[self.cats.index(c1),self.cats.index(c2), self.cats.index(c3)] for c1,c2,c3 in syn_cats])
        strings = torch.LongTensor([[self.toVocabIndex(s1),self.toVocabIndex(s2),self.toVocabIndex(s3)] for s1,s2,s3 in strings])
        
        return {'adj_list':adj_list, 'node_order':node_order, 'sem_embs':sem_embs, 'syn_cats':syn_cats, 'strings':strings, 'root_index':torch.LongTensor([0])}


