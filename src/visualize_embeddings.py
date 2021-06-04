from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

all_embeddings = []
all_words = []

# embeddings_path = "all_word_embeddings.csv"
# embeddings_path = "all_word_embeddings_average_layer2.csv"
embeddings_path = "all_word_embeddings_fastword.csv"
with open(embeddings_path, encoding="utf-8") as word_embedding_file:
    for line in word_embedding_file:
        if "fastword" in embeddings_path:
            w, lang, embedding = line.strip().split("\t")
        else:
            w, embedding = line.strip().split("\t")
        all_words.append(w)
        all_embeddings.append(np.array(eval(embedding)))


def plotWithLabels(denseEmbeddings, labels, filename):
    assert denseEmbeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18)) # in inches
    for i, label in enumerate(labels):
        x, y = denseEmbeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    plt.savefig(filename)


print('visualizing embeddings...')
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
num_to_plot = 500
all_embeddings = np.array(all_embeddings)
print(np.shape(all_embeddings))
dense_embeddings = tsne.fit_transform(all_embeddings[:num_to_plot, :])
labels = all_words[:num_to_plot]
plotWithLabels(dense_embeddings, labels, './tsne.png')
