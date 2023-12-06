import csv
import itertools
import re
import nltk
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


def data_process():
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
        sentences[len(sentences) - 1] += (" %s" % pick_end_token)
        print(len(sentences))

        tokenize_sentences=[nltk.word_tokenize(sent) for sent in sentences]
        maxlen=len(max(tokenize_sentences, key=len))
        word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        vocab = word_freq.most_common()
        index_to_word = [x[0] for x in vocab]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        print("The least picked card is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Create the training data
        x_train = []
        y_train = []
        for sent in tokenize_sentences:
            words=[]
            for w in sent:
                words.append(word_to_index[w])
            x_train.append(words[:-1])
            y_train.append(words[1:])
        x_train=np.array(x_train)
        y_train=np.array(y_train)
    return x_train, y_train, maxlen, len(word_freq), len(sentences)

x_train, y_train, maxlen, dict_size, batch_size = data_process()
seq_len = maxlen - 1

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that card
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

x_train = one_hot_encode(x_train, dict_size, seq_len, batch_size)

x_train = torch.from_numpy(x_train)
y_train = torch.Tensor(y_train)

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

def test(model):
    model.train(False)
    totloss, nbatch = 0., 0
    out, hidden = model(x_train)
    loss = criterion(out,y_train.view(-1).long())
    totloss += loss.item()
    nbatch += 1
    totloss /= float(nbatch)
    model.train(True)
    return totloss

# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=5, n_layers=42)

# Define hyperparameters
n_epochs = 500
lr=0.001

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

totloss_list=[]
testloss_list=[]
instances=[i for i in range(n_epochs)]


# Training Run
for epoch in range(1, n_epochs + 1):
    testloss=test(model)
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output, hidden = model(x_train)
    loss = criterion(output, y_train.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    testloss_list.append(testloss)
    totloss_list.append(loss.item())

    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("testloss: {:.4f}".format(testloss))
        print("loss: {:.4f}".format(loss.item()))

plt.plot(instances, testloss_list)
plt.plot(instances, totloss_list)
plt.show()
