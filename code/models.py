
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.weight_norm import weight_norm

from do_conv import *
from attention import *
from Attn_Net_Gated import *

from einops import rearrange


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

class DOGMA(nn.Module):
    def __init__(self, **config):
        super(DOGMA, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_seq_embedding=config["DRUG"]["EMBEDDING_DIM"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]


        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.drug_extractor_2 = drugconv2d(drug_seq_embedding, num_filters, kernel_size, drug_padding)
        self.protein_extractor = Proteinconv2d(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.gate1 = Attn_Net_Gated(L=128, D=64)
        self.attention=attention(num_head=2, num_hid=256)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d,bg_d_2, v_p, mode="train"):

        v_d = self.drug_extractor(bg_d)
        v_d_2=self.drug_extractor_2(bg_d_2)
        v_p = self.protein_extractor(v_p)
        v_d = self.gate1(v_d)
        v_d_2= self.gate1(v_d_2)
        v_p = self.gate1(v_p)

        f=self.attention(v_d,v_d_2,v_p)
        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d,v_d_2, v_p, f, score
        elif mode == "eval":
            return v_d,v_d_2, v_p, score, None


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class Proteinconv2d(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(Proteinconv2d, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = DOConv2d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm2d(in_ch[1])
    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = rearrange(v, "B C (H W) -> B C H W", H=30)
        v = self.bn1(F.relu(self.conv1(v)))
        v = rearrange(v, "B C H W -> B (H W) C ")
        return v
        
class drugconv2d(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(drugconv2d, self).__init__()
        if padding:
            self.embedding = nn.Embedding(65, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(65, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        self.conv1 = DOConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = rearrange(v, "B C (H W) -> B C H W", H=290)
        v = self.bn1(F.relu(self.conv1(v)))
        v = rearrange(v, "B C H W -> B H (W C)")
        return v


