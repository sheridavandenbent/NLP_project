# coding: utf-8

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--model', type=str,  default='model.pt', help='model to use')
parser.add_argument('--dict', type=str,  default='dict', help='word map to use')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
parser.add_argument('--agreement', type=str, help='location of the test data')
parser.add_argument('--disagreement', type=str, help='location of the test data')
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


    for line in data:
        hidden = model.init_hidden(1)

        input_data = Variable(line["sentence"], volatile=False)
        test_data = line["test"]
        target_data = line["target"]

        output, hidden = model(input_data, hidden)
        output_flat = output.view(-1, ntokens)
        logit = output[-1, :]
        sm = softmax(logit).view(ntokens)

        output_winner = test_data[0]
        output_prob = sm[test_data[0]].data[0]
        if sm[test_data[1]].data[0] > output_prob:
            output_winner = test_data[1]
            output_prob = sm[test_data[1]].data[0]

        if output_winner == target_data[0]:
            correct += 1

        print(tensor_to_sentence(line["sentence"], dictionary))

        print('%s: %f' % (dictionary.idx2word[test_data[0]], sm[test_data[0]].data[0]))
        print('%s: %f' % (dictionary.idx2word[test_data[1]], sm[test_data[1]].data[0]))

    print(correct/total)
    return

if __name__ == '__main__':
    # test sentence and words to check
    # Load dictionary with word ids, load model
    dictionary = pickle.load(open(args.dict, 'rb'), encoding='latin1')
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

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

    total = []
    for val in agreement:
        for line in agreement[val]:
            total.append(line)

    for val in disagreement:
        for line in disagreement[val]:
            total.append(line)

    evaluate(model, dictionary, total)
