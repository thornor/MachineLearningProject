import csv
import itertools
import re
import nltk
import numpy as np


pick_start_token = "PICK_START"
pick_end_token = "PICK_END"

draft_idx=""
print("Reading CSV file...")
with open('Data/draft_data_public.WOE.TradDraft.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader_list=[]
    for x in reader:
        x_re = [re.sub(pattern=" ", repl="_", string=x[i]) for i in range(len(x))]
        x_re = [re.sub(pattern=",", repl="", string=x_re[i]) for i in range(len(x_re))]
        x_re = [re.sub(pattern="'", repl="", string=x_re[i]) for i in range(len(x_re))]
        x_re = [re.sub(pattern="-", repl="", string=x_re[i]) for i in range(len(x_re))]
        reader_list.append(x_re)
    sentences = []
    idx_draft_column = reader_list[0].index("draft_id")
    idx_pick_column = reader_list[0].index("pick")
    for k in range(0,len(reader_list)-1):
        if reader_list[k][idx_draft_column] != reader_list[k+1][idx_draft_column]:
            if len(sentences) != 0:
                sentences[len(sentences)-1] += (" %s" % pick_end_token)
            sentences.append("%s %s" % (pick_start_token, reader_list[k+1][idx_pick_column]))
        else:
            old_sentence = sentences[len(sentences)-1]
            sentences[len(sentences)-1] = ("%s %s" % (old_sentence, reader_list[k+1][idx_pick_column]))
    # pick_order = itertools.chain([nltk.sent_tokenize(x[0].decode('utf-8').lower() for x in reader)])
    sentences[len(sentences) - 1] += (" %s" % pick_end_token)
    print(len(sentences))

    tokenize_sentences=[nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))
    vocab = word_freq.most_common()
    index_to_word = [x[0] for x in vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print("The least picked card is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenize_sentences[0])
    # Create the training data
    x_train = [[word_to_index[w] for w in sent[:-1]] for sent in tokenize_sentences]
    y_train = [[word_to_index[w] for w in sent[1:]] for sent in tokenize_sentences]
    print(x_train[0])
    print("x: \n %s \n %s" % (sentences[0], x_train[0]))
    print("y: \n %s \n %s" % (sentences[0], y_train[0]))

