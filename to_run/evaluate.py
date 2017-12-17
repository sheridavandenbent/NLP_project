# coding: utf-8

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import os
import sys
import torch.nn.functional as F
import numpy
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--model', type=str,  default='model.pt', help='model to use')
parser.add_argument('--dict', type=str,  default='dict', help='word map to use')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
parser.add_argument('--agreement', type=str, help='location of the test data')
parser.add_argument('--disagreement', type=str, help='location of the test data')
parser.add_argument('--other', type=str, help='location of the test data')
args = parser.parse_args()

# function to transform sentence into word id's and put them in a
# pytorch Variable
# NB Assumes the sentence is already tokenised!
def tokenise_words(words, dictionary):
    ids = torch.LongTensor(len(words))
    token = 0
    for word in words:
        try:
            ids[token] = dictionary.word2idx[word]
        except KeyError:
            ids[token] = dictionary.word2idx['<unk>']
        token += 1
    return ids

# softmax function
softmax = nn.Softmax()

def tensor_to_sentence(tensor, dictionary):
    sentence = ""
    for data in tensor:
        sentence += dictionary.idx2word[data]+" "
    return sentence


def evaluate(model, dictionary, data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = dictionary.ntokens()
    correct = 0
    total = len(data)

    logistic_reg = []

    for line in data:
        hidden = model.init_hidden(1)
        ntokens_line = len(line["sentence"])

        input_data = Variable(line["sentence"], volatile=False)
        test_data = line["test"]
        target_data = line["target"]

        output, hidden = model(input_data, hidden)
        output_flat = output.view(-1, ntokens)
        logit = output[-1, :]
        sm = softmax(logit).view(ntokens)

        output_winner = test_data[0]
        output_prob = sm[test_data[0]].data[0]
        output_class = 0
        if sm[test_data[1]].data[0] > output_prob:
            output_winner = test_data[1]
            output_prob = sm[test_data[1]].data[0]
            output_class = 1

        logistic_reg.append({"output":hidden[0][1], "predict":output_class})

        if output_winner == target_data[0]:
            correct += 1
        #else:
            #print(tensor_to_sentence(line["sentence"], dictionary))
            #print('%s: %f' % (dictionary.idx2word[test_data[0]], sm[test_data[0]].data[0]))
            #print('%s: %f' % (dictionary.idx2word[test_data[1]], sm[test_data[1]].data[0]))
            #print('target: %s' % dictionary.idx2word[target_data[0]])


    #print(correct/total)
    return logistic_reg

dictionary = pickle.load(open(args.dict, 'rb'), encoding='latin1')
with open(args.model, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)

ntokens = dictionary.ntokens()
agreement = dict()
with open(args.agreement, 'r') as f:
    cur_val = 0
    for line in f:
        if line[0].isdigit():
            agreement[line[0]] = []
            cur_val = line[0]
        else:
            chunks = line.split(';')
            word_tokens = tokenise_words(chunks[0].lower().split(), dictionary)
            test_tokens = tokenise_words(chunks[1].lower().split('/'), dictionary)
            target_token = tokenise_words(chunks[2].lower().split(), dictionary)
            agreement[cur_val].append({"sentence":word_tokens, "test":test_tokens, "target":target_token})

disagreement = dict()
with open(args.disagreement, 'r') as f:
    cur_val = 0
    for line in f:
        if line[0].isdigit():
            disagreement[line[0]] = []
            cur_val = line[0]
        else:
            chunks = line.split(';')
            word_tokens = tokenise_words(chunks[0].lower().split(), dictionary)
            test_tokens = tokenise_words(chunks[1].lower().split('/'), dictionary)
            target_token = tokenise_words(chunks[2].lower().split(), dictionary)
            disagreement[cur_val].append({"sentence":word_tokens, "test":test_tokens, "target":target_token})

other = []
with open(args.other, 'r') as f:
    for line in f:
        chunks = line.split(';')
        word_tokens = tokenise_words(chunks[0].lower().split(), dictionary)
        test_tokens = tokenise_words(chunks[1].lower().split('/'), dictionary)
        target_token = tokenise_words(chunks[2].lower().split(), dictionary)
        other.append({"sentence":word_tokens, "test":test_tokens, "target":target_token})

total = []
agreement_total = []
disagreement_total = []
other_total = other
for val in agreement:
    for line in agreement[val]:
        total.append(line)
        agreement_total.append(line)

for val in disagreement:
    for line in disagreement[val]:
        total.append(line)
        disagreement_total.append(line)

for line in other:
    total.append(line)

without_other = agreement_total
for val in disagreement:
    for line in disagreement[val]:
        without_other.append(line)


logistic_reg = evaluate(model, dictionary, without_other)
random.shuffle(logistic_reg)

class Model(torch.nn.Module):

    def __init__(self, ntokens):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(ntokens, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

#take 80/20 split
logistic_split = int(len(logistic_reg)/5*4)

logistic_train = logistic_reg[0:logistic_split]
logistic_test = logistic_reg[logistic_split:len(logistic_reg)-1]


nhiddens = 1500
model = Model(nhiddens)

lr = 0.001

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr)

ntrain = len(logistic_train)
logistic_output_list = []
logistic_target_list = []

for data in logistic_train:
    logistic_output_list.append([x for x in data["output"].data[0].view(1,ntokens)[0]])
    logistic_target_list.append([data["predict"]])

logistic_tensor_output = torch.Tensor(logistic_output_list)
logistic_tensor_target = torch.Tensor(logistic_target_list)


for epoch in range(1000):
    #print(epoch)
    y_pred = model(Variable(logistic_tensor_output))
    loss = criterion(y_pred, Variable(logistic_tensor_target))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print(loss.data[0])


#done with training.
#now lets test!
model.eval()
correct = 0
total = len(logistic_test)
for data in logistic_test:
    y_pred = model(Variable(data["output"].data[0].view(1,ntokens))).data[0][0]
    target_class = data["predict"]

    y_class = 0
    if y_pred > 0.5:
        y_class = 1

    if target_class == y_class:
        correct += 1

print(correct/total)
