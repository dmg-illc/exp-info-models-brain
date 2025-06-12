import torch
from transformers import AutoModel, AutoTokenizer
from src.paths import ROOT
from src.utils import *
import pandas as pd
import os
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition', choices=['words']) 
parser.add_argument('-n', '--namespec')  
args = parser.parse_args()

df = pd.read_csv(ROOT / 'data/exp_features.csv')
word_list = df.word.tolist()
model_name = os.path.basename(__file__).split('.')[0]
result_path = ROOT / f'results/{model_name}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)

inputs = tokenizer(word_list, padding=True, truncation=True, return_tensors="pt")
att_mask = inputs.attention_mask.detach().cpu().numpy()

with torch.no_grad():
    embeddings = model(**inputs.to(device), output_hidden_states=True, return_dict=True)
    n_layers = len(embeddings.hidden_states)
    hidden_states = [embeddings.hidden_states[i].detach().cpu().numpy() for i in range(n_layers)] 
    avg_hidden_states = [np.stack([hidden_states[i][j, att_mask[j]==1,:].mean(axis=0) for i in range(n_layers)]) for j in range(len(word_list))]
    out_dict = {word:emb for word,emb in zip(word_list, avg_hidden_states)}


if not os.path.exists(result_path):
    os.mkdir(result_path)

pickle.dump(out_dict, open(result_path / f'{model_name}_{args.namespec}.pkl', "wb"))