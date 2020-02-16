from comet_ml import Experiment
experiment = Experiment(api_key="",
                        project_name="", workspace="")


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import random
import os
import numpy as np
from collections import Counter
import pickle
import json

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import parse_snips, serialize_voc
from dataset import SNIPSDataset
from a_blstm import Encoder, Decoder


lr=0.001
emb_size=128
h_size=128
batch_size=32
epochs = 51
ser_dir = ''
save_checkp = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp = {'lr':0.001,
'emb_size':128,
'h_size':128,
'batch_size':32,
'epochs':51,
'dropout': 0.1}

experiment.log_parameters(hp)

def train(data, data_valid, word2index, tag2index, intent2index):
    dataset = SNIPSDataset(data, word2index, tag2index, intent2index)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dataset_valid = SNIPSDataset(data_valid, word2index, tag2index, intent2index)
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = Encoder(len(word2index), emb_size, h_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index) // 3, h_size * 2)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.init_weights()
    decoder.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim = optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(epochs):

        losses = []
        f1s = []
        precisions = []

        recalls = []
        aps = []

        for i, batch in enumerate(data_loader):
            x, y_slot, y_int = batch

            # print(x)
            # print(len(x))
            # x = torch.cat(x)
            # y_slot = torch.cat(y_slot)
            # y_int = torch.cat(y_int)

            x_mask = torch.cat(
                [Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).to(device) for t in x]).view(
                batch_size, -1)
            y_slot_mask = torch.cat(
                [Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).to(device) for t in y_slot]).view(
                batch_size, -1)

            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x, x_mask)
            start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * batch_size])).to(device).transpose(1, 0)

            slot_score, intent_score, non_modified_score = decoder(start_decode, hidden_c, output, x_mask)

            # l = output.size(1)
            # x_o = start_decode.size(0)
            non_modified_score = torch.argmax(non_modified_score, dim=2)
            predicted_flatten = non_modified_score.view(non_modified_score.shape[0] * non_modified_score.shape[1])
            truth_flatten = y_slot.view(y_slot.shape[0] * y_slot.shape[1])
            # if epoch == 0 and i == 0:
            #     print("Y given",y_slot.shape)
            #     print("Y predicted", non_modified_score.shape)
            #     print("Addd", l, x_o)
            #     print(kek.shape, kek1.shape)

            loss_1 = loss_function_1(slot_score, y_slot.view(-1))
            loss_2 = loss_function_2(intent_score, y_int.view(-1))

            f1_sc = f1_score(truth_flatten.data.to('cpu'), predicted_flatten.data.to('cpu'), average='macro')
            prec = precision_score(truth_flatten.data.to('cpu'), predicted_flatten.data.to('cpu'), average='macro')
            rec = recall_score(truth_flatten.data.to('cpu'), predicted_flatten.data.to('cpu'), average='macro')

            ap = accuracy_score(truth_flatten.data.to('cpu'), predicted_flatten.data.to('cpu'))

            precisions.append(prec)
            f1s.append(f1_sc)
            recalls.append(rec)
            aps.append(ap)

            loss = loss_1 + loss_2
            losss = loss.cpu().item() if torch.cuda.is_available else loss.item()
            losses.append(losss)
            loss.backward()

            torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()
            if i % 100 == 0:
                print("Epoch: {}, batch: {}, loss: {}, f1_score: {}, accuracy: {}".format(epoch, i, losss, f1_sc, ap))
                # losses=[]
            experiment.log_metric("loss", losss)
            experiment.log_metric("acc", ap)
            experiment.log_metric("f1", f1_sc)
            experiment.log_metric("precision", prec)
            experiment.log_metric("recall", rec)

        encoder.eval()
        decoder.eval()
        for j, batch_v in enumerate(data_loader_valid):
            x_val, y_slot_val, y_int_val = batch_v
            x_mask_val = torch.cat(
                [Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).to(device) for t in x_val]).view(
                batch_size, -1)
            y_slot_mask_val = torch.cat(
                [Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).to(device) for t in y_slot_val]).view(
                batch_size, -1)

            start_decode_val = Variable(torch.LongTensor([[word2index['<SOS>']] * batch_size])).to(device).transpose(1,
                                                                                                                     0)

            output_val, hidden_c_val = encoder(x_val, x_mask_val)
            slot_score_val, intent_score_val, non_modified_score_val = decoder(start_decode_val, hidden_c_val,
                                                                               output_val, x_mask_val)

            non_modified_score_val = torch.argmax(non_modified_score_val, dim=2)
            predicted_flatten_val = non_modified_score_val.view(
                non_modified_score_val.shape[0] * non_modified_score_val.shape[1])
            truth_flatten_val = y_slot_val.view(y_slot_val.shape[0] * y_slot_val.shape[1])

            loss_1 = loss_function_1(slot_score_val, y_slot_val.view(-1))
            loss_2 = loss_function_2(intent_score_val, y_int_val.view(-1))

            f1_sc_val = f1_score(truth_flatten_val.data.to('cpu'), predicted_flatten_val.data.to('cpu'),
                                 average='macro')
            prec_val = precision_score(truth_flatten_val.data.to('cpu'), predicted_flatten_val.data.to('cpu'),
                                       average='macro')
            rec_val = recall_score(truth_flatten_val.data.to('cpu'), predicted_flatten_val.data.to('cpu'),
                                   average='macro')

            ap_val = accuracy_score(truth_flatten_val.data.to('cpu'), predicted_flatten_val.data.to('cpu'))
            loss_val = loss_1 + loss_2
            loss_val = loss.cpu().item() if torch.cuda.is_available else loss.item()

            experiment.log_metric("loss_val", loss_val)
            experiment.log_metric("acc_val", ap_val)
            experiment.log_metric("f1_val", f1_sc_val)
            experiment.log_metric("precision_val", prec_val)
            experiment.log_metric("recall_val", rec_val)

        print(f"Validation loss: {loss_val}, f1-score: {f1_sc_val}, accuracy: {ap_val}")
        encoder.train()
        decoder.train()
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(), f'{save_checkp}/encoder_{epoch}.pth')
            torch.save(decoder.state_dict(), f'{save_checkp}/decoder_{epoch}.pth')

        experiment.log_metric("loss_ep", losss)
        experiment.log_metric("acc_ep", ap)
        experiment.log_metric("f1_ep", f1_sc)
        experiment.log_metric("precision_ep", prec)
        experiment.log_metric("recall_ep", rec)

        experiment.log_metric("loss_ep_val", loss_val)
        experiment.log_metric("acc_ep_val", ap_val)
        experiment.log_metric("f1_ep_val", f1_sc_val)
        experiment.log_metric("precision_val", prec_val)
        experiment.log_metric("recall_ep_val", rec_val)




if __name__ == "__main__":
        # make no changes here

        data, word2index, tag2index, intent2index, max_len = parse_snips('./data/snips/train')
        data_valid, _, _, _, _ = parse_snips('./data/snips/valid')
        os.makedirs(ser_dir, exist_ok=True)
        serialize_voc(word2index, ser_dir, 'w2i.pickle')
        serialize_voc(tag2index, ser_dir, 't2i.pickle')
        serialize_voc(intent2index, ser_dir, 'i2i.pickle')
        os.makedirs(save_checkp, exist_ok=True)
        train(data, data_valid, word2index, tag2index, intent2index)
