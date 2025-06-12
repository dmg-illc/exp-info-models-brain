import torch
from transformers import BertModel, AutoTokenizer
from src.paths import ROOT
from src.utils import *
from src.emb_extraction_utils import *
from templates import *
import pandas as pd
import os
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--templates_type', choices=['context', 'visual_context_general', 'visual_context_specific']) 
args = parser.parse_args()

if args.templates_type == 'context':
    current_templates = templates
elif args.templates_type == 'visual_context_general':
    current_templates = gerneral_caption_like_templates
elif args.templates_type == 'visual_context_specific':
    current_templates = per_category_templates
else: 
    raise ValueError('Invalid template type!')

# df = pd.read_csv(ROOT / 'data/exp_features.csv')
df = pd.read_csv(ROOT / 'data/word_categories_2.csv')
word_to_category = {word: category for word, category in zip(df.word.tolist(), df.category.tolist())}
word_list = df.word.tolist()
model_name = os.path.basename(__file__).split('.')[0]
result_path = ROOT / f'results/{model_name}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa").to(device)

# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)

out_dict = {}

with torch.no_grad():

    for word in word_list:

        out_dict[word] = {}
        # input_sentences = [template.format(word=word) for template in templates]
        if type(current_templates) == dict:
            specific_templates = current_templates[word_to_category[word]]
            input_sentences = get_input_sentences(specific_templates, word)
        else:
            input_sentences = get_input_sentences(current_templates, word)
        inputs = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)
        off_map = inputs['offset_mapping'].cpu().numpy()
        del inputs['offset_mapping']
        
        embeddings = model(**inputs.to(device), output_hidden_states=True, return_dict=True).hidden_states
        emb_np = [emb.detach().cpu().numpy() for emb in embeddings]
        
        for i, input_sentence in enumerate(input_sentences):
            indices = get_relevant_tokens_indices(off_map[i], input_sentence, word)
            out_dict[word][f'template_{i+1}'] = np.stack([emb_np[layer][i, indices, :].mean(axis=0) for layer in range(len(emb_np))])

        # break



if not os.path.exists(result_path):
    os.mkdir(result_path)

pickle.dump(out_dict, open(result_path / f'{model_name}_layers_{args.templates_type}.pkl', "wb"))