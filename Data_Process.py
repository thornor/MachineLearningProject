import csv
import itertools
import re
import nltk

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
    print(sentences)

    tokenize_sentences=[nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))