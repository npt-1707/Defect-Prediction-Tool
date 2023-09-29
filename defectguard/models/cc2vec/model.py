import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

# The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hidden_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Word Encoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hidden_size, bidirectional=True)
        # Word Attention
        self.wordattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The sentence RNN model for generating a hunk vector
class SentRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(inp, hid_state)

        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The hunk RNN model for generating the vector representation for the instance
class HunkRNN(nn.Module):
    def __init__(self, hunk_size, hidden_size):
        super(HunkRNN, self).__init__()
        # Sentence Encoder
        self.hunk_size = hunk_size
        self.hunkRNN = nn.GRU(hunk_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.hunkattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.hunkRNN(inp, hid_state)

        hunk_annotation = self.hunkattn(out_state)
        attn = F.softmax(self.attn_combine(hunk_annotation), dim=1)

        hunk = attention_mul(out_state, attn)
        return hunk, hid_state

# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, params):
        super(HierachicalRNN, self).__init__()
        self.params = params
        self.vocab_size = params['vocab_code']
        self.batch_size = params['batch_size']
        self.embed_size = params['embedding_size']
        self.hidden_size = params['cc2vec_hidden_size']
        self.cls = params['cc2vec_class_num']

        self.dropout = nn.Dropout(params['dropout_rate'])  # drop out

        # Word Encoder
        self.wordRNN = WordRNN(self.vocab_size, self.embed_size, self.batch_size, self.hidden_size)
        # Sentence Encoder
        self.sentRNN = SentRNN(self.embed_size, self.hidden_size)
        # Hunk Encoder
        self.hunkRNN = HunkRNN(self.embed_size, self.hidden_size)

        # standard neural network layer
        self.standard_nn_layer = nn.Linear(self.embed_size * 2, self.embed_size)

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(self.embed_size, self.embed_size)
        self.W_nn_tensor_two = nn.Linear(self.embed_size, self.embed_size)
        self.V_nn_tensor = nn.Linear(self.embed_size * 2, 2)

        # Hidden layers before putting to the output layer
        self.fc1 = nn.Linear(3 * self.embed_size + 4, 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.cls)
        self.sigmoid = nn.Sigmoid()

    def forward_code(self, x, hid_state, device='cpu'):
        hid_state_hunk, hid_state_sent, hid_state_word = hid_state
        n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        # i: hunk; j: line; k: batch
        hunks = None
        for i in range(n_hunk):
            sents = None
            for j in range(n_line):
                words = [x[k][i][j] for k in range(n_batch)]
                words = np.array(words)
                sent, state_word = self.wordRNN(torch.tensor(words, device=device).view(-1, self.batch_size), hid_state_word)
                sents = sent if sents is None else torch.cat((sents, sent), 0)
            hunk, state_sent = self.sentRNN(sents, hid_state_sent)
            hunks = hunk if hunks is None else torch.cat((hunks, hunk), 0)
        hunks = torch.mean(hunks, dim=0)  # hunk features
        return hunks

    def forward(self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word, device='cpu'):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        x_added_code = self.forward_code(x=added_code, hid_state=hid_state, device=device)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state, device=device)

        x_added_code = x_added_code.view(self.batch_size, self.embed_size)
        x_removed_code = x_removed_code.view(self.batch_size, self.embed_size)

        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        return torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=1)

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(self.batch_size, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(self.batch_size, 1)

    def standard_neural_network_layer(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.standard_nn_layer(concat)
        output = F.relu(output)
        return output

    def neural_network_tensor_layer(self, added_code, removed_code):
        output_one = self.W_nn_tensor_one(removed_code)
        output_one = torch.mul(output_one, added_code)
        output_one = torch.sum(output_one, dim=1).view(self.batch_size, 1)

        output_two = self.W_nn_tensor_two(removed_code)
        output_two = torch.mul(output_two, added_code)
        output_two = torch.sum(output_two, dim=1).view(self.batch_size, 1)

        W_output = torch.cat((output_one, output_two), dim=1)
        code = torch.cat((removed_code, added_code), dim=1)
        V_output = self.V_nn_tensor(code)
        return F.relu(W_output + V_output)

    def init_hidden_hunk(self, device='cpu'):
        return torch.zeros((2, self.batch_size, self.hidden_size), device=device)

    def init_hidden_sent(self, device):
        return torch.zeros((2, self.batch_size, self.hidden_size), device=device)

    def init_hidden_word(self, device):
        return torch.zeros((2, self.batch_size, self.hidden_size), device=device)
    
class DeepJITExtended(nn.Module):
    def __init__(self, params):
        super(DeepJITExtended, self).__init__()
        self.params = params

        V_msg = params['vocab_msg']
        V_code = params['vocab_code']
        Dim = params['embedding_size']
        Class = params['deepjit_class_num']
        Embedding = params['embedding_feature']

        Ci = 1  # input of convolutional layer
        Co = params['num_filters']  # output of convolutional layer
        Ks = params['filter_sizes']  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(params['dropout_rate'])
        self.fc1 = nn.Linear(2 * len(Ks) * Co + Embedding, params['deepjit_hidden_size'])  # hidden units
        self.fc2 = nn.Linear(params['deepjit_hidden_size'], Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.params['num_filters'] * len(self.params['filter_sizes']))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, ftr, msg, code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((ftr, x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out