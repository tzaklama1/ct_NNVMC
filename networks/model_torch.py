# model.py
import torch
import torch.nn as nn

class CoTrainingTransformer(nn.Module):
    """
    Input  : occupation sequence (0/1) of length N
    Params : (t, V)  -> passed as two special tokens
    Architecture:
      ① token+pos embed ② self-attn encoder
      ③ cross-attn where queries≈site states, keys/values≈param tokens
    Output : scalar log-amplitude  log|ψ(σ; t,V)|
    """
    def __init__(self, N_sites: int, d_model: int = 256,
                 n_layers: int = 8, n_heads: int = 8):
        super().__init__()
        self.site_emb  = nn.Embedding(2, d_model)          # 0/1 occupancy
        self.pos_emb   = nn.Parameter(torch.randn(1, N_sites, d_model))

        self.param_proj = nn.Linear(2, d_model)            # (t,V) → token
        enc_layer = nn.TransformerEncoderLayer(d_model,
                                               n_heads,
                                               4*d_model,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        # --- cross-attention block ---
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                batch_first=True)

        # --- read-out ---
        self.head = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.SiLU(),
                                  nn.Linear(d_model, 1))

    def forward(self, occ, params):
        """
        occ    : (B, N)   int  {0,1}
        params : (B, 2)   float  (t, V)
        """
        B, N = occ.shape
        x = self.site_emb(occ) + self.pos_emb               # (B,N,D)

        p_tok = self.param_proj(params).unsqueeze(1)        # (B,1,D)
        # ① site-only contextualisation
        h = self.encoder(x)                                 # (B,N,D)
        # ② cross-attend (sites <- params)
        h2, _ = self.cross_attn(query=h, key=p_tok, value=p_tok)
        h = h + h2                                          # residual
        # ③ global pool (mean)
        h_mean = h.mean(dim=1)
        return self.head(h_mean).squeeze(-1)                # (B,)
