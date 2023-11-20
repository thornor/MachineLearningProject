import csv
import itertools
import nltk

pick_start_token = "PICK_START"
pick_end_token = "PICK_END"

draft_idx=""

print("Reading CSV file...")
with open('Data/draft_data_public.WOE.TradDraft.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    # idx_draft_column = reader.index("draft_id")
    # idx_card_column = reader.index("pick")
    pick_order = itertools.chain([nltk.sent_tokenize(x[0].decode('utf-8').lower() for x in reader)])
    print(pick_order)