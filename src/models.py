from pytorch_metric_learning import losses, miners
import torch.nn as nn
import torch
import math
from transformers import BertForMaskedLM, AutoModelForMaskedLM


# Comtrastive learning model with one transformation layer
class ContrastiveLearningModel(nn.Module):
    def __init__(self, in_dim, out_dim, infonce_tau=0.04):
        super(ContrastiveLearningModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.infoNCE_tau = infonce_tau
        # NTXentLoss is infoNCE loss, it will generate negative pairs automatically
        self.loss = losses.NTXentLoss(temperature=self.infoNCE_tau)
        self.trans = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.miner = miners.MultiSimilarityMiner()
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, in_mv1, in_mv2, use_hard_pair=False, labels=None):
        # in_mv1: first items of positive pairs
        # in_mv2: second items of positive pairs

        # A.dot(X)
        transformed_mv1 = self.trans(in_mv1)
        transformed_mv2 = self.trans(in_mv2)

        embedding_all = torch.cat([transformed_mv1, transformed_mv2], dim=0)
        embedding_all = self.bn(embedding_all)

        if labels is None:
            labels = torch.arange(transformed_mv1.size(0))
        labels = torch.cat([labels, labels], dim=0)

        if use_hard_pair:
            hard_pairs = self.miner(embedding_all, labels)
            loss = self.loss(embedding_all, labels, hard_pairs)
        else:
            loss = self.loss(embedding_all, labels)
        return loss

    def transformation(self, input):
        transformed_mv = self.trans(input)
        return transformed_mv


# Comtrastive learning model with two transformation layer
class ContrastiveLearningModel2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, infonce_tau=0.04, nonlinear=False):
        super(ContrastiveLearningModel2, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.infoNCE_tau = infonce_tau
        self.nonlinear = nonlinear
        # NTXentLoss is infoNCE loss, it will generate negative pairs automatically
        self.loss = losses.NTXentLoss(temperature=self.infoNCE_tau)
        self.trans1 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.trans2 = nn.Linear(self.hidden_dim, self.out_dim, bias=True)
        self.miner = miners.MultiSimilarityMiner()
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, in_mv1, in_mv2, use_hard_pair=False, labels=None):
        # in_mv1: first items of positive pairs
        # in_mv2: second items of positive pairs

        # A.dot(X)
        hidden_mv1 = self.trans1(in_mv1)
        hidden_mv2 = self.trans1(in_mv2)

        if self.nonlinear:
            hidden_mv1 = self.relu(hidden_mv1)
            hidden_mv2 = self.relu(hidden_mv2)

        transformed_mv1 = self.trans2(hidden_mv1)
        transformed_mv2 = self.trans2(hidden_mv2)

        embedding_all = torch.cat([transformed_mv1, transformed_mv2], dim=0)
        embedding_all = self.bn(embedding_all)

        if labels is None:
            labels = torch.arange(transformed_mv1.size(0))
        labels = torch.cat([labels, labels], dim=0)

        if use_hard_pair:
            hard_pairs = self.miner(embedding_all, labels)
            loss = self.loss(embedding_all, labels, hard_pairs)
        else:
            loss = self.loss(embedding_all, labels)
        return loss

    def transformation(self, input):
        hidden_mv = self.trans1(input)
        transformed_mv = self.trans2(hidden_mv)
        return transformed_mv


# CNN model
class CNN_Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, cnn_kernel_size, vec_num_per_word, stride=1, padding=0, pool_way='max'):
        super(CNN_Classifier, self).__init__()

        self.in_dim = in_dim
        self.cnn_out_channel = out_dim
        self.cnn_in_channel = vec_num_per_word
        self.cnn_kernel_size = cnn_kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = out_dim
        self.pool_way = pool_way

        self.conv = nn.Conv1d(in_channels=self.cnn_in_channel,
                              out_channels=self.cnn_out_channel,
                              kernel_size=self.cnn_kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=1,
                              groups=1)
        if self.pool_way == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=self.pool_kernel_size, stride=1)
        else:
            self.pooling = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=1)

        self.cls_in_dim = math.floor((self.in_dim + 2 * self.padding - (self.cnn_kernel_size - 1) - 1) / self.stride + 1)
        self.cls = nn.Linear(self.cls_in_dim, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.sigmoid_fn = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, target=None):
        # input: w_n, vn_per_w, in_dim
        h1 = self.conv(input)  # w_n, self.out_dim, self.cls_in_dim
        h1 = h1.permute(0, 2, 1)  # w_n, self.cls_in_dim, self.out_dim
        h2 = self.pooling(h1)  # w_n, self.cls_in_dim, 1
        logits = self.cls(h2.squeeze(2)).squeeze(1)
        if target is not None:
            loss = self.loss_fn(logits, target)
            return logits, loss
        return logits


# Simple BERT masked model
class BertMaskModel(torch.nn.Module):
    def __init__(self, config, bert_version='bert-large-uncased-whole-word-masking'):
        super(BertMaskModel, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)
        self.miner = miners.MultiSimilarityMiner()
        if config.loss == "infonce":
            self.loss_fn = losses.NTXentLoss(temperature=config.tau)
        elif config.loss == "triple":
            self.loss_fn = losses.TripletMarginLoss(margin=config.margin)
        self.use_hard_pair = config.use_hard_pair

    def forward(self, token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, labels=None):

        emb_in_1 = self.get_bert_emb(token_ids_1, input_mask_1, indices_1)
        emb_in_2 = self.get_bert_emb(token_ids_2, input_mask_2, indices_2)

        emb_all = torch.cat([emb_in_1, emb_in_2], dim=0)

        if labels is None:
            labels = torch.arange(emb_in_1.size(0))
        labels = torch.cat([labels, labels], dim=0)

        if self.use_hard_pair:
            hard_pairs = self.miner(emb_all, labels)
            loss = self.loss_fn(emb_all, labels, hard_pairs)
        else:
            loss = self.loss_fn(emb_all, labels)
        return loss

    def get_bert_emb(self, token_ids, input_masks, indices):
        last_layer_emb = self.bert(input_ids=token_ids, attention_mask=input_masks).hidden_states[-1]
        sent_ids = torch.tensor(list(range(last_layer_emb.shape[0])), dtype=torch.long)
        hidden_mask = last_layer_emb[sent_ids, indices]
        del token_ids
        del input_masks
        del indices
        del last_layer_emb
        del sent_ids

        return hidden_mask


# BERT masked model with MLP
class BertMaskModelWithMLP(torch.nn.Module):
    def __init__(self, config, bert_version='bert-large-uncased-whole-word-masking'):
        super(BertMaskModelWithMLP, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)
        self.in_dim = self.bert.config.hidden_size
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        self.miner = miners.MultiSimilarityMiner()
        self.nonlinear = config.nonlinear
        if config.loss == "infonce":
            self.loss_fn = losses.NTXentLoss(temperature=config.tau)
        elif config.loss == "triple":
            self.loss_fn = losses.TripletMarginLoss(margin=config.margin)
        self.use_hard_pair = config.use_hard_pair

        self.trans1 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.trans2 = nn.Linear(self.hidden_dim, self.out_dim, bias=True)

        self.relu = nn.ReLU()

    def forward(self, token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, labels=None):
        # bert input vectors
        emb_in_1 = self.get_bert_emb(token_ids_1, input_mask_1, indices_1)
        emb_in_2 = self.get_bert_emb(token_ids_2, input_mask_2, indices_2)

        # hidden states of first transformation layer
        h1_1 = self.trans1(emb_in_1)
        h1_2 = self.trans1(emb_in_2)

        # if nonlinear, apply relu to first hidden layer
        if self.nonlinear:
            h1_1 = self.relu(h1_1)
            h1_2 = self.relu(h1_2)

        # hidden states of second transformation layer
        h2_1 = self.trans2(h1_1)
        h2_2 = self.trans2(h1_2)

        # concatenation
        emb_all = torch.cat([h2_1, h2_2], dim=0)

        if labels is None:
            labels = torch.arange(emb_in_1.size(0))
        labels = torch.cat([labels, labels], dim=0)

        if self.use_hard_pair:
            hard_pairs = self.miner(emb_all, labels)
            loss = self.loss_fn(emb_all, labels, hard_pairs)
        else:
            loss = self.loss_fn(emb_all, labels)
        return loss

    def get_bert_emb(self, token_ids, input_masks, indices):
        last_layer_emb = self.bert(input_ids=token_ids, attention_mask=input_masks).hidden_states[-1]
        sent_ids = torch.tensor(list(range(last_layer_emb.shape[0])), dtype=torch.long)
        hidden_mask = last_layer_emb[sent_ids, indices]
        del token_ids
        del input_masks
        del indices
        del last_layer_emb
        del sent_ids

        return hidden_mask