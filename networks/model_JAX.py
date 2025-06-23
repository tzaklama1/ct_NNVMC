# model.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int = 8
    def setup(self):
        self.attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0
        )
        self.mlp  = nn.Sequential([
            nn.Dense(4*self.d_model), nn.gelu,
            nn.Dense(self.d_model),
        ])
        self.ln1  = nn.LayerNorm()
        self.ln2  = nn.LayerNorm()

    def __call__(self, x, *, train: bool):
        h = self.ln1(x)
        x = x + self.attn(h, deterministic=not train)
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x

class CrossAttention(nn.Module):
    d_model: int
    n_heads: int = 8
    def setup(self):
        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.out_proj = nn.Dense(self.d_model)

    def __call__(self, q, kv):
        Q, K, V = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
        h = nn.attention.dot_product_attention(
            Q, K, V,
            bias=None, dropout_rate=0.0, deterministic=True
        )
        return self.out_proj(h)

class CotrainTransformer(nn.Module):
    """
    Maps (occ_seq, params) → complex coefficient (Re, Im)
    occ_seq : int[batch, N]   (0/1)
    params  : float[batch, 2] (t,V)
    """
    n_sites: int
    d_model: int = 256
    depth: int   = 8
    n_heads: int = 8

    def setup(self):
        # Embeddings
        self.occ_embed = nn.Embed(num_embeddings=2, features=self.d_model)
        self.pos_embed = self.param('pos', nn.initializers.normal(stddev=0.02),
                                    (1, self.n_sites, self.d_model))
        self.param_proj = nn.Dense(self.d_model)

        # Encoder stack
        self.blocks = [SelfAttentionBlock(self.d_model, self.n_heads)
                       for _ in range(self.depth)]

        # Cross-attention
        self.cross = CrossAttention(self.d_model, self.n_heads)

        # Read-out head
        self.head = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.d_model), nn.gelu,
            nn.Dense(2)                # → (Re, Im)
        ])

    def __call__(self, occ, params, *, train: bool = False):
        """
        occ    : (B,N) int32
        params : (B,2) float32
        """
        x = self.occ_embed(occ) + self.pos_embed           # (B,N,D)
        for blk in self.blocks:
            x = blk(x, train=train)

        # Prepare params tokens (B,1,D)
        p_tok = self.param_proj(params)[..., None, :]      # unsqueeze length=1

        # Cross-attend: queries = sites, keys/values = params
        x = x + self.cross(x, p_tok)

        # Pool to global rep by mean
        g = x.mean(axis=1)                                 # (B,D)
        coeff = self.head(g)                               # (B,2)
        return coeff                                       # complex(Re,Im)
