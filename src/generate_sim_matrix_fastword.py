import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description='Generate a similarity matrix using edit distance or word embeddings')
parser.add_argument('--embeddings_file', type=str, nargs='?', default='all_word_embeddings_fastword.csv')
parser.add_argument('--out_file', type=str, nargs='?', default='sim_matrix_fastword')

options = parser.parse_args()

max_dist = 0

all_words = []

word_to_embedding = {}
word_to_id = {}
with open(options.embeddings_file, encoding="utf-8") as word_embedding_file:
    for i, line in enumerate(word_embedding_file):
        w, lang, embedding = line.strip().split("\t")
        word_to_embedding[w, lang] = np.array(eval(embedding))
        word_to_id[w + '\t' + lang] = i

        all_words.append((w, lang))

min_sim = None


def word_embedding_similarity(w1, w2, l1=None, l2=None):
    global min_sim, max_dist
    e1 = word_to_embedding[w1, l1]
    e2 = word_to_embedding[w2, l2]
    mag_e1 = np.sqrt(np.sum(e1 ** 2))
    mag_e2 = np.sqrt(np.sum(e2 ** 2))
    cos_sim = (e1.dot(e2)) / (mag_e1 * mag_e2)

    # euclid_dist = np.sqrt(np.sum(np.square(e1 - e2)))

    if min_sim is None or cos_sim < min_sim:
        min_sim = cos_sim

    # if euclid_dist > max_dist:
        # max_dist = euclid_dist

    return cos_sim
    # return euclid_dist


sim_matrix = np.zeros((len(all_words), len(all_words)))

for i in range(len(all_words)):
    w1, lang1 = all_words[i]
    l1_key = word_to_id[w1 + '\t' + lang1]
    for j in range(i, len(all_words)):
        w2, lang2 = all_words[j]
        l2_key = word_to_id[w2 + '\t' + lang2]
        sim_val = word_embedding_similarity(w1, w2, l1=lang1, l2=lang2)
        sim_matrix[l1_key][l2_key] = sim_val

        sim_matrix[l2_key][l1_key] = sim_val

    if i % 100 == 0:
        print(i, flush=True)

print("similarity matrix created!", flush=True)

# normalize to 0-1 scale and convert edit distance to a similarity metric
print("normalizing....", flush=True)
print("minimum cos sim value:", min_sim)
sim_matrix = (sim_matrix - min_sim) / (1 - min_sim)

modifier = ""
modifier += "_word_embeddings" + "_cos_sim"  # "_euclidian"

with open(options.out_file + modifier + "_key.json", "w", encoding="utf-8") as out_file:
    json.dump(word_to_id, out_file)

with open(options.out_file + modifier + ".npy", "wb") as out_file:
    np.save(out_file, sim_matrix)
