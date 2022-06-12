from load import *
import torch
from torch import nn
from torch.nn import functional as F


import math

seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda'


def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


# class AttnMatch(nn.Module):
#     def __init__(self, emb_loc, loc_max, dropout=0.1):
#         super(AttnMatch, self).__init__()
#         self.value = nn.Linear(max_len, 1, bias=False)
#         self.emb_loc = emb_loc
#         self.loc_max = loc_max
#
#     def forward(self, self_attn, self_delta, traj_len):
#         # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
#         self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension
#         [N, L, M] = self_delta.shape
#         candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
#         candidates = candidates.unsqueeze(0).expand(N, -1).to(device)  # (N, L)
#         emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
#         attn = torch.bmm(emb_candidates, self_attn.transpose(-1, -2))  # (N, L, M)
#         # attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)E sanjiaox
#         # pdb.set_trace() M+
#         attn_out = self.value(attn).view(N, L)  # (N, L) /GEN d
#         # attn_out = F.log_softmax(attn_out, dim=-1)  # ignore if cross_entropy_loss
#
#         return attn_out  # (N, L)


class AttnMatch(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.1):
        super(AttnMatch, self).__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max

    def forward(self, self_attn, mat2, traj):
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        #self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension
        [N, M, T] = traj.shape
        [L,tmp]=mat2.shape
        candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.bmm(emb_candidates, self_attn.transpose(-1, -2)) # (N, L, M)
        #attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)
        # pdb.set_trace() M+
        attn_out = self.value(attn).view(N, L)  # (N, L) /GEN d
        #attn_out = F.log_softmax(attn_out, dim=-1)  # ignore if cross_entropy_loss

        return attn_out  # (N, L)



# class Embed(nn.Module):
#     def __init__(self, ex, emb_size, loc_max, embed_layers):
#         super(Embed, self).__init__()
#         _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
#         self.su, self.sl, self.tu, self.tl = ex
#         self.emb_size = emb_size
#         self.loc_max = loc_max
#
#     def forward(self, traj_loc, mat2, vec, traj_len):
#         # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
#         delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
#         delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
#         mask = torch.zeros_like(delta_t, dtype=torch.long)
#         for i in range(mask.shape[0]):  # N
#             mask[i, 0:traj_len[i]] = 1
#             delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])
#
#         # pdb.set_trace()
#
#
#         esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
#         vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
#                              (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
#                              (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
#                              (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
#
#         space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
#         time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
#         delta = space_interval + time_interval  # (N, M, L, emb)
#
#         return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
        self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(traj[:, :, 2])  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1])  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0])  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)


        return joint#1,100,256


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        mask = torch.sum(mask, -1)  # squeeze the embed dimension
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))#最后一维

       # scores=query

        if mask is not None:
  #scores torch.Size([1, 8, 100, 100])
  #mask torch.Size([1, 100, 100, 256])

            scores = scores.masked_fill(mask == 0, -1e9)#torch.Size([1, 8, 100, 32])

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn#zhangliangxiangcheng


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer embed_dim？？
        :param attn_heads: head sizes of multi-head attention embed_dim？？
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate +
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, traj_len):
        #  joint  traj/delta
        mask = torch.zeros_like(x, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h#32
        self.h = h#8

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)#全连接层
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        #print(d_model)
        #print(self.d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 4.2, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))