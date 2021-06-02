from utils.this_queue_sim_matrix import OurQueue
from collections import defaultdict  # , Counter
from scipy.sparse import load_npz, save_npz, csr_matrix, find
from scipy.io import mmwrite
# from dataio import save_folds, save_weak_folds
# from itertools import product
# from math import log
import pandas as pd
import numpy as np
import argparse
import time
import os
# import copy
import json
# import sys
# import resource
# import gc
# from collections import Counter

# custom_windows = [60*60*24 * 4, 60*60*24 * 2, 60*60*12]
# custom_windows = []
custom_windows = None
NB_TIME_WINDOWS = len(custom_windows) + 1 if custom_windows is not None else 5

parser = argparse.ArgumentParser(description='Prepare data for DAS3H using a similarity matrix')
parser.add_argument('sim_file', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy_tw')
parser.add_argument('--tags', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--lemma', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--subword_skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--vocab_size', type=int, nargs='?', default=500)
parser.add_argument('--nbest', type=int, nargs='?', default=10)
parser.add_argument('--continuous_correct', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--continuous_wins', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--bias', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--tw', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--pfa', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--l1', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def df_to_sparse(full, q_matrix, sim_matrix=None, sim_key=None, skill_to_word=dict(), tw=True, pfa=False, users=False, items=False, skills=False, l1=False, verbose=True):
    dt = time.time()
    # full = copy.deepcopy(full)
    # full.sort_values(by="timestamp", inplace=True)

    if 'skill_id' in full.columns:
        full['skill_id'] = full['skill_id'].astype(pd.Int64Dtype())  # Can be NaN

    nb_samples, _ = full.shape
    shift_skills = 0
    if full['user_id'].dtype == np.int64:  # We shift IDs to ensure that
        shift_items = 1 + full['user_id'].max()  # user/item/skill IDs are distinct
        full['item_id_original'] = full['item_id']
        full['item_id'] += shift_items
        shift_skills = int(1 + full['item_id'].max())

    # Handle skills (either q-matrix, or skill_id, or skill_ids from 0)
    q_mat = defaultdict(list)
    # all_skill_ids_by_lang = dict()
    skill_id_to_sim_id_cache = dict()
    # skill_id_to_lang = dict()

    nb_skills = None
    if 'skill_id' in full.columns:
        print('Found a column skill_id')
        full['skill_id'] += shift_skills
    else:
        # elif os.path.isfile('q_mat.npz'):
        #     print('Found a q-matrix')
        #     q_matrix = load_npz('q_mat.npz')
        _, nb_skills = q_matrix.shape
        rows, cols, _ = find(q_matrix)
        for i, j in zip(rows, cols):
            q_mat[shift_items + i].append(shift_skills + j)

            skill_id = int(j)

            # if skill_id in skill_id_to_lang:
            #     skill_lang = skill_id_to_lang[skill_id]
            # else:
            #     skill_lang = skill_to_word[skill_id].split(":", 1)[0]
            #     skill_id_to_lang[skill_id] = skill_lang

            if skill_id not in skill_id_to_sim_id_cache:
                skill_lang, skill_string = skill_to_word[skill_id].split(":", 1)
                skill_word = skill_string.split("/", 1)[0]
                if skill_word == "<*sf>":
                    skill_word = skill_string.split("/", 1)[1].split("<", 1)[0]
                word_index = sim_key[skill_word + "\t" + skill_lang]
                skill_id_to_sim_id_cache[skill_id] = word_index

            # if skill_lang not in all_skill_ids_by_lang:
            #     all_skill_ids_by_lang[skill_lang] = set()
            # all_skill_ids_by_lang[skill_lang].add(skill_id)

    del q_matrix

    shift_langs = int(1 + nb_skills + shift_skills)
    full['combined_lang_id'] += shift_langs

    full['i'] = range(nb_samples)
    print('Loading data:', nb_samples, 'samples', time.time() - dt, flush=True)
    print(full.head(), flush=True)

    all_values = {}
    if nb_skills is None:
        nb_skills = 112  # Only way to know for Algebra 2005

    conversion = {}
    if users:
        conversion['user_id'] = 'user'
    if items:
        conversion['item_id'] = 'item'
    if skills:
        conversion['skill_id'] = 'kc'
    if l1:
        conversion['combined_lang_id'] = 'l1'

    for col in conversion:
        if col in full.columns:
            all_values[col] = full[col].dropna().unique()
            all_values[col] = all_values[col][all_values[col].argsort()]
        else:
            all_values['skill_id'] = list(range(shift_skills,
                                                shift_skills + nb_skills))

    # Create folds of indices and save them
    # if not os.path.isfile('folds/{}fold0.npy'.format(nb_samples)):
    #     save_folds(full)
    # save_folds(full)

    # Preprocess codes
    dt = time.time()
    codes = dict(zip([value for field, key in conversion.items()
                      for value in all_values[field]], range(1000000)))
    print('Preprocess codes', time.time() - dt, flush=True)

    for field, key in conversion.items():
        print("field:", field, "start:", codes[all_values[field][0]], "end:", codes[all_values[field][-1]], flush=True)


    # Extra codes for counters within time windows (wins, attempts)
    if options.tw:
        extra_codes = dict(zip([(field, value, pos)
                                for field in ['wins', 'attempts']
                                for value in all_values['skill_id']
                                for pos in range(NB_TIME_WINDOWS)],
                               range(len(codes), len(codes) + 1000000)))
    else:
        extra_codes = dict()
    print('Gather all', len(codes) + len(extra_codes), 'features', flush=True)

    matrix_shape_1 = len(codes) + len(extra_codes)

    convert = np.vectorize(codes.get)
    for field, key in conversion.items():
        dt = time.time()
        if field != 'skill_id':  # Will not work because of potential NaN values
            full[key] = convert(full[field])
            print('Encode', key, time.time() - dt, flush=True)

    dt = time.time()
    rows = []
    cols = []
    data = []

    if users:
        rows += list(range(nb_samples))
        cols += list(full['user'])
        data += [1] * (nb_samples)
    if items:
        rows += list(range(nb_samples))
        cols += list(full['item'])
        data += [1] * (nb_samples)
    if l1:
        rows += list(range(nb_samples))
        cols += list(full['l1'])
        data += [1] * (nb_samples)

    assert len(rows) == len(cols) == len(data)
    print('Initialized', len(rows), 'entries', time.time() - dt, flush=True)

    def add(r, c, d):
        rows.append(r)
        cols.append(c)
        data.append(d)

    def identity(x):
        return x

    suffix = 'uis'
    if tw:
        suffix = 'das3h'
        link_function = lambda x: np.log(1 + x)
    elif pfa:
        suffix = 'swf'
        link_function = identity

    # var_list = list(locals().items()) + list(globals().items())
    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in var_list),
    #                          key= lambda x: -x[1])[:20]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)), flush=True)
    #
    # print("total number of variables:", len(var_list), flush=True)
    # print("total memory used by this process:", resource.getrusage(resource.RUSAGE_SELF).ru_idrss, flush=True)
    # os.system("free -m")
    # c = Counter(type(o) for o in gc.get_objects())
    # print(c.most_common(20), flush=True)

    if tw or pfa:  # Build time windows features
        df = full
        if 'skill_id' in full.columns:
            # df = df.dropna(subset=['skill_id'])
            df['skill_ids'] = df['skill_id'].astype(str)
        else:
            df['skill_ids'] = [None] * len(df)

        del full

        dt = time.time()
        # Prepare counters for time windows
        q = defaultdict(lambda: OurQueue(only_forever=pfa, custom_windows=custom_windows))
        # Using zip is the fastest way to iterate DataFrames
        # Source: https://stackoverflow.com/a/34311080
        print("total lines to process:", len(df["i"]), flush=True)
        count = 0

        if sim_matrix is None or sim_key is None or not options.continuous_wins:
            raise Exception("sim_matrix or sim_key missing or continuous_wins is false")

        for i_sample, user, item_id, t, correct, skill_ids in zip(
               df['i'], df['user_id'], df['item_id'], df['timestamp'], df['correct'],
               df['skill_ids']):
            # print("i_sample", i_sample)
            # print("user", user)
            # print("item_id", item_id)
            # print("t", t)
            # print("correct", correct)
            # print("skill_ids", skill_ids)
            for skill_id in q_mat[item_id] or skill_ids.split('~~'):  # Fallback
                skill_id = int(skill_id)
                shifted_skill_id = skill_id - shift_skills

                # l_skill1 = skill_id_to_lang[shifted_skill_id]
                w1_index = skill_id_to_sim_id_cache[shifted_skill_id]

                if skills:
                    add(i_sample, codes[skill_id], 1)

                total_counters, correct_counters = q[user].get_counters(t, w1_index, sim_matrix)
                for pos, (total_value, correct_value) in enumerate(zip(total_counters, correct_counters)):
                    if total_value > 0:
                        # print("attempts counter", value, flush=True)
                        add(i_sample, extra_codes['attempts', skill_id, pos],
                            link_function(total_value))
                    if correct_value > 0:
                        # print("attempts counter", value, flush=True)
                        add(i_sample, extra_codes['wins', skill_id, pos],
                            link_function(correct_value))

                q[user].push(t, w1_index, correct=correct)

            count += 1
            if count % 10000 == 0:
                print(count, "lines processed", flush=True)
                # var_list = list(locals().items()) + list(globals().items())
                # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in var_list),
                #                          key= lambda x: -x[1])[:20]:
                #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                #
                # print("total number of variables:", len(var_list), flush=True)
                # print("total memory used by this process:", resource.getrusage(resource.RUSAGE_SELF).ru_idrss, flush=True)
                # os.system("free -m")
                # c = Counter(type(o) for o in gc.get_objects())
                # print(c.most_common(20), flush=True)
                # print("----", flush=True)

        print('Run all', len(q), 'queues', time.time() - dt)
        print('Total', len(rows), 'entries')
        print("Done processing!", flush=True)

    col_offset = 7
    cols = list(np.array(cols) + col_offset)

    rows += list(range(len(df))) * col_offset
    for i in range(col_offset):
        cols += [i] * len(df)
    data += list(df["user_id"]) + list(df["item_id_original"]) + list(df["timestamp"]) + list(df["correct"]) + list(df["inter_id"]) + list(df["session_correct"]) + list(df["session_seen"])

    matrix_shape_1 += col_offset

    if options.bias:
        rows += list(range(len(df)))
        cols += [matrix_shape_1] * len(df)
        data += [1] * len(df)
        matrix_shape_1 += 1

    dt = time.time()
    X = csr_matrix((data, (rows, cols)), shape=(len(df), matrix_shape_1))

    print("matrix created!", flush=True)

    return X

    # y = np.array(full['correct'])
    # print('Into sparse matrix', X.shape, y.shape, time.time() - dt)
    # dt = time.time()
    # save_npz('X-{}.npz'.format(suffix), X)
    # np.save('y-{}.npy'.format(suffix), y)
    # print('Saving', time.time() - dt)

# dt = time.time()
# os.chdir('data/{}'.format(options.dataset))
# full = pd.read_csv('needed.csv')  # Only 176.7 MB for ASSISTments 2012 (3 GB)
# # full = pd.read_csv('preprocessed_data.csv',sep="\t")


if __name__ == "__main__":
    os.chdir(os.path.join('data', options.dataset))

    all_features = ['users', 'items', 'skills', 'l1']
    active_features = [features for features in all_features if vars(options)[features]]
    features_suffix = ''.join([features[0] for features in active_features])
    features_suffix += 'wa'
    if options.tw:
        features_suffix += 't1'

    #LIST_OF_BOUNDARIES = [1/24, 1, 7, 30, np.inf]

    modifier = ""
    if options.subword_skills:
        submodifier = ""
        if options.tags:
            submodifier += "_tags"
        if options.lemma:
            submodifier += "_lemma"
        modifier = "_subword"+submodifier+"_vocab="+str(options.vocab_size)+"_nbest="+str(options.nbest)
    elif options.tags:
        modifier = "_tags"
        if options.lemma:
            modifier += "_lemma"
    elif options.lemma:
        modifier = "_lemma"

    if options.continuous_correct:
        modifier += "_continuous"

    df = pd.read_csv('preprocessed_data'+modifier+'.csv', sep="\t")
    qmat = load_npz('q_mat'+modifier+'.npz').toarray()

    skill_to_word = {}
    with open("skill_map" + modifier + ".csv", encoding="utf-8") as skill_map_file:
        for line in skill_map_file:
            skill_id, skill_string = line.strip().split("\t")
            skill_to_word[int(skill_id)] = skill_string

    with open(options.sim_file + "_key.json", "r", encoding="utf-8") as key_file:
        word_to_id = json.load(key_file)

    with open(options.sim_file + ".npy", "rb") as matrix_file:
        sim_matrix = np.load(matrix_file)
    X = df_to_sparse(df, qmat, sim_matrix=sim_matrix, sim_key=word_to_id, skill_to_word=skill_to_word, tw=options.tw, users=options.users, items=options.items, skills=options.skills, l1=options.l1)

    if options.continuous_wins and options.tw:
        modifier += "_continuous_wins"

    if custom_windows is not None and options.tw:
        modifier += "_windows=" + str(custom_windows)

    if not options.bias:
        modifier += "_no_bias"

    modifier += "_" + options.sim_file.split("/")[-1]

    out_file = 'X'+modifier+'-{:s}.npz'.format(features_suffix)
    save_npz(out_file, X)
    # mmwrite(out_file.replace(".npz", ""), X)

    print("matrix saved! all done!", flush=True)
