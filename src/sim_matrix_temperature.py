from scipy.sparse import load_npz, save_npz, csr_matrix, find
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Generate a similarity matrix using edit distance or word embeddings')
parser.add_argument('--sim_file', type=str, nargs='?', default='sim_matrix')
parser.add_argument('--cosine', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--euclid', type=bool, nargs='?', const=True, default=False)

options = parser.parse_args()

assert options.cosine or options.euclid

with open(options.sim_file + "_key.json", "r", encoding="utf-8") as key_file:
    word_to_id = json.load(key_file)

with open(options.sim_file + ".npy", "rb") as matrix_file:
    sim_matrix = np.load(matrix_file)


if options.cosine:
    temperatures = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # temperatures = np.linspace(0.0, 1.0, num=11)
    # print(temperatures, flush=True)
    # temperatures = temperatures[1:-1] # already have results for 0.0 and 1.0
elif options.euclid:
    temperatures = np.linspace(0.2, 2.0, num=10)


di = np.diag_indices(sim_matrix.shape[0])

for t in range(len(temperatures)):
    current_temp = temperatures[t]
    print("starting temp", current_temp, "...", flush=True)

    sim_matrix_t = copy.deepcopy(sim_matrix)
    if options.cosine:
        sim_matrix_t *= current_temp
        sim_matrix_t[di] = 1
    elif options.euclid:
        sim_matrix_t = np.exp(-current_temp * sim_matrix_t)

    with open(options.sim_file + "_temp=" + str(current_temp) + "_key.json", "w", encoding="utf-8") as out_file:
        json.dump(word_to_id, out_file)

    with open(options.sim_file + "_temp=" + str(current_temp) + ".npy", "wb") as out_file:
        np.save(out_file, sim_matrix_t)

    print("done with temp", current_temp, "!", flush=True)
