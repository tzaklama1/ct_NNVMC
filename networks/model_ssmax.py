# Module with scalable softmax => for large lattices
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

def scalable_softmax(logits: jnp.ndarray,
                     axis: int = -1,
                     s: jnp.ndarray | float = 0.5) -> jnp.ndarray:
    """Scalable-Softmax (Nakanishi 2025, Eq.(2))."""
    n = logits.shape[axis]
    scale = s * jnp.log(float(n))
    logits = logits * scale
    logits = logits - jnp.max(logits, axis=axis, keepdims=True)  # stability
    exp   = jnp.exp(logits)
    return exp / jnp.sum(exp, axis=axis, keepdims=True)

# ---- REPLACE ResidualSelfAttention WITH ------------------------------------
class ResidualSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    mlp_dims: tuple[int, ...]
    ln: bool = True
    def setup(self):
        h = self.d_model // self.n_heads
        self.wq = nn.Dense(self.d_model, use_bias=False)
        self.wk = nn.Dense(self.d_model, use_bias=False)
        self.wv = nn.Dense(self.d_model, use_bias=False)
        self.wo = nn.Dense(self.d_model, use_bias=False)
        self.s  = self.param('sscale', lambda k: jnp.array(0.5))  # learnable s
        self.mlp = [nn.Dense(m) for m in self.mlp_dims] + \
                   [nn.Dense(self.d_model)]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, *, train: bool):
        q = self.wq(self.norm1(x))
        k = self.wk(x)
        v = self.wv(x)
        # reshape to (B, heads, L, h)
        B,L,_ = q.shape
        q = q.reshape(B, L, self.n_heads, -1).transpose(0,2,1,3)
        k = k.reshape(B, L, self.n_heads, -1).transpose(0,2,1,3)
        v = v.reshape(B, L, self.n_heads, -1).transpose(0,2,1,3)
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(k.shape[-1])
        attn   = scalable_softmax(logits, axis=-1, s=self.s)
        out    = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out    = out.transpose(0,2,1,3).reshape(B, L, -1)
        x = x + self.wo(out)                     # residual-1
        h = self.norm2(x)
        for layer in self.mlp[:-1]:
            h = nn.gelu(layer(h))
        x = x + self.mlp[-1](h)                  # residual-2
        return x

# ---- REPLACE CrossAttentionParams WITH -------------------------------------
class CrossAttentionParams(nn.Module):
    d_model: int
    n_heads: int
    def setup(self):
        self.wq = nn.Dense(self.d_model, use_bias=False)
        self.wk = nn.Dense(self.d_model, use_bias=False)
        self.wv = nn.Dense(self.d_model, use_bias=False)
        self.wo = nn.Dense(self.d_model, use_bias=False)
        self.s  = self.param('sscale_cross', lambda k: jnp.array(0.5))

    def __call__(self, sites, params_tok):
        B,L,_ = sites.shape
        q = self.wq(sites).reshape(B,L,self.n_heads,-1).transpose(0,2,1,3)
        k = self.wk(params_tok).reshape(B,1,self.n_heads,-1).transpose(0,2,1,3)
        v = self.wv(params_tok).reshape(B,1,self.n_heads,-1).transpose(0,2,1,3)
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(k.shape[-1])
        attn   = scalable_softmax(logits, axis=-1, s=self.s)
        out    = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out    = out.transpose(0,2,1,3).reshape(B, L, -1)
        return self.wo(out)
