
import os
import pickle

def parse_snips(path, test=False, pickle_folder=None, w2i=None, t2i=None, i2i=None):
    with open(os.path.join(path, 'seq.in'), 'r') as file:
        text = file.readlines()
    text = [t[:-1] for t in text]

    with open(os.path.join(path, 'seq.out'), 'r') as file:
        slots = file.readlines()

    # slots = [t.replace("\n", '') for t in slots]
    slots = [t[:-2] if " \n" in t else t[:-1] for t in slots]

    with open(os.path.join(path, 'label'), 'r') as file:
        intents = file.readlines()

    data = [[text[i].split(' '), slots[i].split(' '), intents[i]] for i in range(len(text))]

    seq_in, seq_out, intent = list(zip(*data))

    word_vocab = set([elem for sent in seq_in for elem in sent])
    slot_vocab = set([elem for sent in seq_out for elem in sent])
    intent_vocab = set(intent)

    lengths = [len(elem) for elem in seq_in]

    max_len = max(lengths)

    if not test:
        seq_in_n = []
        seq_out_n = []

        for i in range(len(seq_in)):
            temp = seq_in[i]
            if len(temp) < max_len:
                temp.append('<EOS>')
                while len(temp) < max_len:
                    temp.append('<PAD>')
            seq_in_n.append(temp)

            temp = seq_out[i]
            if len(temp) < max_len:
                while len(temp) < max_len:
                    temp.append('<PAD>')
            seq_out_n.append(temp)

        seq_in = seq_in_n
        seq_out = seq_out_n

    data = list(zip(seq_in, seq_out, intent))

    if test:
        if pickle_folder is None:
            raise ValueError("A path to serialized train vocabularie should be provided in test mode")
        else:
            with open(os.path.join(pickle_folder, "w2i.pickle"), 'rb') as f:
                word2index = pickle.load(f)
            with open(os.path.join(pickle_folder, "t2i.pickle"), 'rb') as f:
                tag2index = pickle.load(f)
            with open(os.path.join(pickle_folder, "i2i.pickle"), 'rb') as f:
                intent2index = pickle.load(f)
        return data, word2index, tag2index, intent2index, 0
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    counter = len(word2index)
    for token in word_vocab:
        if token not in word2index.keys():
            word2index[token] = counter
            counter += 1

    index2word = {v: k for k, v in word2index.items()}

    tag2index = {'<PAD>': 0}
    counter = len(tag2index)
    for tag in slot_vocab:
        if tag not in tag2index.keys():
            tag2index[tag] = counter
            counter += 1
    index2tag = {v: k for k, v in tag2index.items()}

    intent2index = {}
    counter = len(intent2index)
    for ii in intent:
        if ii not in intent2index.keys():
            intent2index[ii] = counter
            counter += 1
    index2intent = {v: k for k, v in intent2index.items()}

    return data, word2index, tag2index, intent2index, max_len


def serialize_voc(voc, path, name):

  with open(os.path.join(path, name), 'wb') as f:

      pickle.dump(voc, f)