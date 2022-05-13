import fasttext

EMBEDDING_SIZE = 300
ft_dir = "../../../fastText/"

all_words = []
ft_by_lang = {}
langs = set()

with open("all_words_langs.csv", encoding="utf-8") as words_file:
    for line in words_file:
        lang, word = line.strip().split("\t")
        langs.add(lang)
        all_words.append((word, lang))

print("loaded all words...")

for lang in langs:
    ft_by_lang[lang] = fasttext.load_model(ft_dir + 'cc.' + lang + ".300.bin")
    print("loaded", lang, "model")

all_words = sorted(all_words)

with open("all_word_embeddings_fastword.csv", "w", encoding="utf-8") as out_file:
    for i, (word, lang) in enumerate(all_words):
        embedding = ft_by_lang[lang][word]

        assert len(embedding) == EMBEDDING_SIZE

        out_file.write(word + "\t" + lang + "\t" + str(list(embedding)) + "\n")
        print(i, "done")
