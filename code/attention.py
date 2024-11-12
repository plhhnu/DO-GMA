

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange

class attention(nn.Module):
    def __init__(self, num_head=8, num_hid=128, dropout=0.2):
        super(attention, self).__init__()
        self.num_hid = num_hid
        self.num_head = num_head
        self.linear_v = nn.Linear(num_hid, num_hid)
        self.linear_k = nn.Linear(num_hid, num_hid)
        self.linear_q = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)


    def att(self,  key, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        if query.size(1) > 1:
            att_map = self.dropout(att_map)

        return  att_map

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)
        
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)  

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)
        att_map1 = self.att(k, q,
                            mask)
        att_map2 = self.att(v, q,
                            mask)
        att_map=att_map1+att_map2
        x=v+k
        y=q
        temp = torch.einsum('bvk,bqv->bqk', (x[:, 0, :, :], att_map[:, 0, :, :]))
        logits = torch.einsum('bqk,bqk->bk', (temp, y[:, 0, :, :]))
        for i in range(1, self.num_head):
            temp = torch.einsum('bvk,bqv->bqk', (x[:, i, :, :], att_map[:, i, :, :]))
            logits_i = torch.einsum('bqk,bqk->bk', (temp, y[:, i, :, :]))
            logits=torch.concat((logits,logits_i),dim=-1)
        att=logits

        return att

