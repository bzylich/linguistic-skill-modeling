import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser(description='Visualize a similarity matrix based on edit distance or word embeddings')
parser.add_argument('--in_file', type=str, nargs='?', default='sim_matrix')
# parser.add_argument('--out_file', type=str, nargs='?', default='sim_matrix')
# parser.add_argument('--edit_distance', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--word_embeddings', type=bool, nargs='?', const=True, default=False)

options = parser.parse_args()

with open(options.in_file + "_key.json", "r", encoding="utf-8") as key_file:
    word_to_id = json.load(key_file)

with open(options.in_file + ".npy", "rb") as matrix_file:
    sim_matrix = np.load(matrix_file)

# sim_matrix = sim_matrix[:50, :50]

plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
plt.show()
