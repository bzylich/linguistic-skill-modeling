# Linguistic Skill Modeling for Second Language Acquisition

This repository contains the Python code used for the experiments from our LAK 2021 paper: [Linguistic Skill Modeling for Second Language Acquisition](https://dl.acm.org/doi/10.1145/3448139.3448153). Authors: Brian Zylich (bzylich@umass.edu) and Andrew Lan.

The data we use can be [downloaded](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) from Duolingo. We also use their [HLR code](https://github.com/duolingo/halflife-regression) as a baseline for our experiments.

Some of the code in this repository is based on code from [Benoît Choffin](https://github.com/BenoitChoffin)'s [DAS3H repository](https://github.com/BenoitChoffin/das3h).

## Running the baselines
Use src/hlr.py to run the baselines from [Duolingo's repository](https://github.com/duolingo/halflife-regression).

'''
python hlr.py path/to/learning_traces.13m.csv -m [MODEL_NAME]
'''

Models include hlr, lr, leitner, and pimsleur.


## Preparing the data for our models

To prepare data for our models, use the following command (using src/das3h/prepare_data.py). Note that the --lemma and --tags options are optional. These correspond to using the lemma form of words and including linguistic tags, respectively. You may also have to change the path to the dataset in the script.

'''
python prepare_data.py --dataset duolingo_hlr --min_interactions 0 --continuous_correct --lemma --tags

'''

If you would like to run our subword model, you will need to use the following options instead. In addition, you must run src/train_tokenizer.py first to generate the subword tokenizers. This also requires that you create a "vocab" subfolder within "src/das3h".

'''
python train_tokenizer.py
python prepare_data.py --dataset duolingo_hlr --min_interactions 0 --continuous_correct --subword_skills --tokenizer_dir ./vocab/ --vocab_size 5000 --nbest 2

'''

If you would like to run our DKT baseline, you should instead use src/dkt/generate_dkt_features.py to prepare the data.
'''
python generate_dkt_features.py --max_seq_len 200
'''

## 
