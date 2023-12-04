import csv
import itertools
import re
import nltk
import numpy as np
import torch
from torch import nn


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

        print("\nExample sentence: '%s'" % sentences[0])
        print("\nExample sentence after Pre-processing: '%s'" % tokenize_sentences[0])
        # Create the training data
        x_train = []
        y_train = []
        for sent in tokenize_sentences:
            words=[]
            for w in sent:
                words.append(word_to_index[w])
            x_train.append(words[:-1])
            y_train.append(words[1:])
            if len(words) != 44:
                print(len(words), sent, tokenize_sentences.index(sent))
        x_train=np.array(x_train)
        y_train=np.array(y_train)
    return x_train, y_train, maxlen, len(word_freq), len(sentences)

x_train, y_train, maxlen, dict_size, batch_size = data_process()
seq_len = maxlen - 1
x_train = np.array(x_train)
y_train = np.array(y_train)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

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
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    x_train.to(device)
    output, hidden = model(x_train)
    loss = criterion(output, y_train.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)