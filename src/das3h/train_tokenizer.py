import sentencepiece as spm
import pandas as pd
import os

vocab_sizes = [5000]

df = pd.read_csv("../../data/learning_traces.13m.csv", encoding="utf-8")[["lexeme_string", "learning_language"]]

surface_form = df["lexeme_string"].apply(lambda x: x.split("/")[0] if "<*sf>" not in x else x.split("/")[1].split("<")[0])
lang_id = df["learning_language"]

lang_words = {}
unique_lang_words = {}
langs = set()

for i in range(len(surface_form)):
    lang = lang_id[i]

    langs.add(lang)

    # if not os.path.exists("./vocab/"+lang+"_text.txt"):
    surface = surface_form[i]

    if lang not in lang_words:
        lang_words[lang] = [surface]
        unique_lang_words[lang] = set([surface])
    else:
        lang_words[lang].append(surface)
        unique_lang_words[lang].add(surface)

unk_id = 0
bos_id = -1
eos_id = -1
pad_id = -1

print("Unique words (surface forms) in each language:", flush=True)
for lang in langs:
    # if not os.path.exists("./vocab/"+lang+"_text.txt"):
    print(lang, ":", len(unique_lang_words[lang]), flush=True)

    with open("./vocab/"+lang+"_text.txt", "w", encoding="utf-8") as lang_file:
        for word in lang_words[lang]:
            lang_file.write(word+"\n")

    in_files = ["./vocab/"+lang+"_text.txt"]
    for vocab_size in vocab_sizes:
        print("creating tokenizer for", lang, "vocab size:", vocab_size, flush=True)
        sp_model_prefix = "./vocab/" + lang+"_sp_model_vocab="+str(vocab_size)
        options = '--input='+','.join(in_files)+' --model_prefix='+sp_model_prefix+' --vocab_size='+str(vocab_size)+' --unk_id='+str(unk_id)+' --bos_id='+str(bos_id)+' --eos_id='+str(eos_id)+' --pad_id='+str(pad_id) + ' --character_coverage=0.995 --hard_vocab_limit=False'

        spm.SentencePieceTrainer.Train(options)
