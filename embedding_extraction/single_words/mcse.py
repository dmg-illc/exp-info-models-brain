# from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import torch
from src.paths import ROOT
from src.utils import *
import pandas as pd
import os
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition', choices=['words']) 
parser.add_argument('-m', '--model', choices=['bert', 'roberta'])
parser.add_argument('-d', '--dataset', choices=['flickr', 'coco'])
parser.add_argument('-n', '--namespec')  

args = parser.parse_args()

df = pd.read_csv(ROOT / 'data/exp_features.csv')
word_list = df.word.tolist()
model_name = os.path.basename(__file__).split('.')[0]
result_path = ROOT / f'results/{model_name}'
lm = 'bert-base-uncased' if args.model=='bert' else 'roberta-base'
model_id = f"UdS-LSV/mcse-{args.dataset}-{lm}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

inputs = tokenizer(word_list, padding=True, truncation=True, return_tensors="pt")
att_mask = inputs.attention_mask.detach().cpu().numpy()


with torch.no_grad():
    embeddings = model(**inputs.to(device), return_dict=True).pooler_output
    out_dict = {word:emb for word,emb in zip(word_list, embeddings.detach().cpu().numpy())}



if not os.path.exists(result_path):
    os.mkdir(result_path)

pickle.dump(out_dict, open(result_path / f'{model_name}_{args.model}_{args.dataset}_{args.namespec}.pkl', "wb"))