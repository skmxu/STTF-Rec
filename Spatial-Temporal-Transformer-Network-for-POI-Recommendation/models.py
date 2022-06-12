# from layers import *
#
#
# class Model(nn.Module):
#     def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout=0.1):
#         super(Model, self).__init__()
#         emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
#         emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
#         emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
#         emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
#         emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
#         emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
#         emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
#         n_layers = 1
#         self.embed_dim=embed_dim
#         embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl
#         self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)#embed_dim 8
#         self.transformer_blocks = nn.ModuleList(
#             [TransformerBlock(embed_dim, 8, embed_dim * 4, dropout) for _ in range(n_layers)])#head
#         self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
#         self.AttnMatch = AttnMatch(emb_l, l_dim-1)
#
#     def forward(self, traj, mat1, mat2, vec, traj_len):
#         x= self.MultiEmbed(traj, mat1, traj_len)  # (1, 100, 256), (N, M, M, emb)
#         for transformer in self.transformer_blocks:
#             x = transformer.forward(x,traj_len)
#         self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)  # (N, M, L, emb)
#         output = self.AttnMatch(x, self_delta, traj_len)  # (N, L)
#         return output
from layers import *


class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        n_layers = 1
        self.embed_dim=embed_dim
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl
        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)#embed_dim 8
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, 8, embed_dim * 4, dropout) for _ in range(n_layers)])#head
        #self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.AttnMatch = AttnMatch(emb_l, l_dim-1)

    def forward(self, traj, mat1, mat2, vec, traj_len):
        x= self.MultiEmbed(traj, mat1, traj_len)  # (1, 100, 256), (N, M, M, emb)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x,traj_len)
        #self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)  # (N, M, L, emb)
        output = self.AttnMatch(x, mat2, traj)  # (N, L)
        return output