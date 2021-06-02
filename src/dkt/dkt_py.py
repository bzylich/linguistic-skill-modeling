import torch
from torch import nn
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, device=device)[y]


def loss_with_z_term(loss_fn, z_hat, y, class_weights=None, seen=None, z_weight=1.0, eps=1e-6):
    y_clamp = torch.clamp(y, eps, 1.0 - eps)
    z = torch.log(y_clamp / (1-y_clamp))

    # y_view = y.view(-1, 1)
    if seen is not None:
        return (loss_fn(z_hat, y).flatten() + z_weight * torch.square(z - z_hat).flatten()) * seen
    else:
        if class_weights is not None:
            weight_indices = torch.floor(y / 0.1).long().view(-1)
            weight_indices[weight_indices == 10] = 9
            # print(weight_indices.size(), flush=True)
            class_weights_to_apply = class_weights[weight_indices]
            # print(class_weights_to_apply.size(), flush=True)
            loss_intermediate = loss_fn(z_hat, y).view(-1) + z_weight * torch.square(z - z_hat).view(-1)
            # print(loss_intermediate.size(), flush=True)
            return loss_intermediate * class_weights_to_apply
        return loss_fn(z_hat, y) + z_weight * torch.square(z - z_hat)


class DKT(nn.Module):
    def __init__(self, n_question, embed_l, n_time_bins, hidden_dim, num_layers=1, class_weights=None, final_fc_dim=512, dropout=0.0, z_weight=0.0, pretrained_embeddings=None, freeze_pretrained=True):
        super().__init__()
        """
        Input:
            n_question : number of concepts + 1. question = 0 is used for padding.
            n_time_bins : number of time bins + 1. bin = 0 is used for padding.
        """
        self.n_question = n_question
        self.n_time_bins = n_time_bins
        self.dropout = dropout
        self.z_weight = z_weight
        self.class_weights = class_weights

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=embed_l + self.n_time_bins + 2,  # word embedding, time bin, session num correct, session num attempted
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + embed_l + self.n_time_bins, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1)
        )

        if pretrained_embeddings is not None:
            # self.reset()
            print("embeddings frozen:", freeze_pretrained, flush=True)
            self.q_embed = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0, freeze=freeze_pretrained)
        else:
            self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)
            # self.reset()

    # def reset(self):
    #     for p in self.parameters():
    #         torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, correct_data, attempts_data, time_bin_data, target, mask):
        """
            input:
            q_data : shape seqlen,  batchsize, concept id, from 1 to NumConcept, 0 is padding
            qa_data : shape seqlen, batchsize, concept response id, from 1 to 2*NumConcept, 0 is padding
            target : shape seqlen, batchsize, -1 is for padding timesteps.
        """
        q_embed_data = self.q_embed(q_data)  # seqlen, BS,   d_model
        time_bin_categorical = to_categorical(time_bin_data, self.n_time_bins)

        batch_size, sl = q_data.size(1), q_data.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        rnn_input = torch.cat([q_embed_data, time_bin_categorical, torch.unsqueeze(correct_data, 2), torch.unsqueeze(attempts_data, 2)], dim=2)
        hidden_seq, _ = self.rnn(rnn_input, h_0)
        # print(h_0.size(), hidden_seq.size(), flush=True)
        h = torch.cat([h_0[-1:, :, :], hidden_seq], dim=0)[:-1, :, :]  # T,BS,hidden_dim
        # print(h.size(), flush=True)
        # print(q_embed_data.size(), time_bin_categorical.size(), flush=True)
        ffn_input = torch.cat([h, q_embed_data, time_bin_categorical], dim=2)  # concatenate time-shifted hidden states with current question info

        pred = self.out(ffn_input)  # Size (Seqlen, BS, n_question+1)
        # pred = pred.view(-1, self.n_question)
        # qs = q_data.view(-1)
        # pred = pred[torch.arange(batch_size*sl, device=device), qs]

        labels = target.view(-1)
        m = nn.Sigmoid()
        preds = pred.view(-1)  # logit

        # mask = labels != -1
        mask = mask.view(-1)
        masked_labels = labels[mask]
        masked_preds = preds[mask]

        loss = nn.BCEWithLogitsLoss(reduction='none')
        # loss = nn.MSELoss()
        # out = loss(masked_preds, masked_labels)
        out = loss_with_z_term(loss, masked_preds, masked_labels, class_weights=self.class_weights, z_weight=self.z_weight)
        return out, m(preds), mask.sum()
        # return out, torch.clamp(preds, 0.0, 1.0), mask.sum()
