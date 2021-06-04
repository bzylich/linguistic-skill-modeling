# Linguistic Skill Modeling for Second Language Acquisition

This repository contains the Python code used for the experiments from our LAK 2021 paper: [Linguistic Skill Modeling for Second Language Acquisition](https://dl.acm.org/doi/10.1145/3448139.3448153). Authors: Brian Zylich (bzylich@umass.edu) and Andrew Lan.

The data we use can be [downloaded](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) from Duolingo. We also use their [HLR code](https://github.com/duolingo/halflife-regression) as a baseline for our experiments.

Some of the code in this repository is based on code from [Benoît Choffin](https://github.com/BenoitChoffin)'s [DAS3H repository](https://github.com/BenoitChoffin/das3h).

## Dependencies
- Python 3
- [PyTorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [sklearn](https://scikit-learn.org/stable/install.html)

## Running the baselines
Use src/hlr.py to run the baselines from [Duolingo's repository](https://github.com/duolingo/halflife-regression).

    python hlr.py path/to/learning_traces.13m.csv -m [MODEL_NAME]

Models include hlr, lr, leitner, and pimsleur.


## Preparing the data for our models
If you plan to run DKT or DAS3H with word embeddings, you must use src/process_fastword_embeddings.py to retrieve all the necessary word embeddings for the dataset. This requires the [fasttext python module](https://pypi.org/project/fasttext/) and the relevant language models from [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html).

    python process_fastword_embeddings.py

To prepare data for our models, use the following command (using src/das3h/prepare_data.py). Note that the --lemma and --tags options are optional. These correspond to using the lemma form of words and including linguistic tags, respectively. You may also have to change the path to the dataset in the script.


    python prepare_data.py --dataset duolingo_hlr --min_interactions 0 --continuous_correct --lemma --tags

If you would like to run our subword model, you will need to use the following options instead. In addition, you must run src/train_tokenizer.py first to generate the subword tokenizers. This also requires that you create a "vocab" subfolder within "src/das3h".


    python train_tokenizer.py
    python prepare_data.py --dataset duolingo_hlr --min_interactions 0 --continuous_correct --subword_skills --tokenizer_dir ./vocab/ --vocab_size 5000 --nbest 2


If you would like to run our neural embedding similarity variant of DAS3H or the DKT baseline, you should instead use src/dkt/generate_dkt_features.py to prepare the data.

    python generate_dkt_features.py --max_seq_len 200

## Encoding features for DAS3H

If you are using DKT or the neural embedding similarity approaches, you can skip this step.

To encode data for our das3h models without word embeddings, use the following command (using src/das3h/fast_encode.py). Note that again the --lemma and --tags options are optional. Note that this script expects the preprocessed data to be in the src/das3h/data/duolingo_hlr/ directory. The outputs will also be saved to this directory.

    python fast_encode.py --dataset duolingo_hlr --tw --continuous_correct --continuous_wins --users --items --skills --l1 --max_history_len 200 --tags --lemma
     
To encode data using subwords use the following command (using src/das3h/fast_encode.py). You can change the vocab_size and nbest options to match what you chose when training the subword tokenizers.

    python fast_encode.py --dataset duolingo_hlr --tw --continuous_correct --continuous_wins --users --items --skills --l1 --max_history_len 200 --subword_skills --vocab_size 5000 --nbest 2 

To encode data using the cosine embedding similarity DAS3H variant, you should use src/das3h/fast_encode_sim_matrix.py. Note that you must first construct a similarity matrix with src/generate_sim_matrix_fastword.py.

    python generate_sim_matrix_fastword.py --embeddings_file all_word_embeddings_fastword.csv --out_file sim_matrix_fastword
    
    python fast_encode_sim_matrix.py sim_matrix_filename_without_file_extension --dataset duolingo_hlr --tw --users --items --skills --l1 --continuous_correct --continuous_wins

## Running our models

To run das3h using the DAS3H variants (except neural embedding similarity), use src/das3h/das3h.py.

    python das3h.py path/to/data/encoded_data_filename.npz --dataset duolingo_hlr --duo_split --continuous_correct --d 0

To run our neural embedding similarity DAS3H variant, use src/das3h/das3h_neural_embeddings.py.

    python das3h_neural_embeddings.py

To run DKT, use src/dkt/train_dkt.py. You can optionally choose to freeze or finetune word embeddings. You can also run without pretrained word embeddings by not setting the embeddings_file option. You can set the hyperparameters that you want to use within the code, and optionally use the grid_search option to perform a hyperparameter search.

    python train_dkt.py --grid_search --embeddings_file path/to/embeddings_all_word_embeddings_fastword.npy --freeze_embeddings
