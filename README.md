# Experiential Semantic Information and Brain Alignment: Are Multimodal Models Better than Language Models?

This repository contains the code base for the homonymous [paper](https://aclanthology.org/2025.conll-1.10/) by Anna Bavaresco and Raquel Fernández, accepted to CoNLL 2025.

![Exp setup](https://github.com/user-attachments/assets/f66b6c87-0a31-4bef-ab2b-46ea933242d8)

## Structure

The content of this repo is structures as follows:

```
.
├── LICENSE
├── README.md
├── data
│   └── README.md
├── embedding_extraction
│   ├── single_words
│   │   ├── bert.py
│   │   ├── clap.py
│   │   ├── mcse.py
│   │   ├── simcse.py
│   │   └── visualbert.py
│   └── words_in_context
│       ├── bert.py
│       ├── clap.py
│       ├── clip.py
│       ├── mcse.py
│       ├── simcse.py
│       ├── templates.py
│       └── visualbert.py
├── requirements.txt
├── rsa
│   ├── partial_correlations_top3_layers.ipynb
│   ├── rsa_all_layers.ipynb
│   ├── rsa_top3_best_layers.ipynb
│   └── statistical_tests.ipynb
├── setup.py
└── src
    ├── __init__.py
    ├── emb_extraction_utils.py
    ├── fmri_rsa_utils.py
    ├── paths.py
    └── utils.py

```
## Getting data

Please look at the README inside the `data` folder for instructions on where to find the fMRI data.

## Setting up 

If you'd like to rerun the code yourself, set your environment up by running the following commands:

```
python -m venv exp-env
source exp-env/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Extracting embeddings

The Python scripts to extract embeddings are included in the `embedding_extraction` folder. Use the files in `embedding_extraction/single_words` if you wish to extract embeddings by passing isolated nouns to the models. If, instead, you'd like to extract contextualised representations, look at the files in `embedding_extraction/words_in_context`.

## Running analyses

Code to reproduce the main analyses conducted as part of our experiments can be found in the `rsa` folder. More specifically, look at:

* `rsa_all_layers.ipynb` if you want to compute RSA for all model layers;

* `rsa_top3_best_layers.ipynb` if you want to compute RSA by averaging representations from the top-3 best layers;

* `partial_correlations_top3_layers.ipynb` if you want to run the partial correlation analysis;

* `rsa/statistical_tests.ipynb` if you want to reproduce our statistical tests.


If you find any on the contents of this repo useful, please consider citing our work:

```
@inproceedings{bavaresco-etal-2024-exp,
    title = "Experiential Semantic Information and Brain Alignment: {A}re Multimodal Models Better than Language Models?",
    author = "Bavaresco, Anna  and
      Fern{\'a}ndez, Raquel",
    booktitle = "Proceedings of the 29th Conference on Computational Natural Language Learning",
    year = "2025"
}

```
