# codi

## Requirements:

pandas
numpy

benepar
spacy
nltk
torch
transformers
sentence_transformers
os
json
pickle
math
argparse
tqdm
time
datetime
random
collections

IsoScore

sklearn

## 1. Installing Dependencies
conda create --name _env_ python=3.9

conda install -n _env_ pip

conda activate _env_

pip install -r requirements.txt

You will need to download the English language model for SpaCy. You can do this by running the following command from the terminal:

`python -m spacy download en_core_web_sm`

You will also need to download the benepar parsing model. You can do this by running the following command from the python terminal:


```python
 import benepar
 benepar.download('benepar_en3')
```

## 2. Train the model

Go to SentRepLearning directory (`src/SentRepLearning`) 

Parse wiki dataset for training (Copy wiki data into data/wiki/)

```
python Preparse.py --dataset wiki --size 100000
```

Train the model with GloVe/RoBERTa/Llama initialization (Copy GloVe model file to data/ before train with GloVe initialization)

```
python IterTrain.py --size 100000 --init\_model glove
python IterTrain.py --size 100000 --init\_model roberta
python IterTrain.py --size 100000 --init\_model llama
```

This script will train and save the model to models/_modelname_ where _modelname_ is `codi`, `codi\_init\_modelroberta`, or `codi\_init\_modelllama` 

## 3. Analyse Compositional Operator

Go to SentRepLearning directory (`src/SentRepLearning`) 

To print syntactic category wise IsoScore and Spread 
```
python Analyse.py --companalyse --model _modelname_
```

To print height wise Norms and IsoScores of semantic embeddings of nodes in the tree
```
python Analyse.py --printscore --model _modelname_
```

## 4. Generate Sentence Representations of Probing Task Datasets

Go to SentRepLearning directory (`src/SentRepLearning`) 

Parse samples in the probing task dataset, _dataset_
```
python Preparse.py --dataset _dataset_
```

Generate representations
```
python GenSentReps.py --dataset _dataset_ --modelname _modelname_ 
```
This will generate the sentence representations of _dataset_ in data/cache


## 5. Evaluate

Go to Evaluation directory (`src/Evaluate`)

Train and evaluate MLP classfiers by using each sentence representations generated in data/cache
```
python Evaluate.py --classifier sklearn --benchmark probing
```

Calculate baseline scores
```
python Baseline.py
```
 
