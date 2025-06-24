# model.py
"""
PsiFormer-style transformer for lattice Fock states.
Input  : occ ∈ {0,1}^{N_sites}, params = (t,V)
Output : complex coefficient  ψ_sigma(t,V) = Re + i Im
"""
from typing import Sequence
import jax, jax.numpy as jnp
import flax.linen as nn

# --------------------------------------------------------------------------- #
#                       Low-level building blocks                             #
# --------------------------------------------------------------------------- #
class ResidualSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    mlp_dims: Sequence[int]
    ln: bool = True

    @nn.compact
    def __call__(self, x, *, train: bool):
        h = nn.LayerNorm()(x) if self.ln else x
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            kernel_init=nn.initializers.lecun_normal(),
            dropout_rate=0.0,
            deterministic=not train
        )(h)
        x = x + h                          # residual

        h = nn.LayerNorm()(x) if self.ln else x
        for dim in self.mlp_dims:
            h = nn.Dense(dim)(h); h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h                       # residual


class CrossAttentionParams(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, sites, params_tok):
        # sites: (B,N,D) queries | params_tok: (B,1,D) keys+values
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=0.0,
            deterministic=True
        )
        return attn(sites, params_tok)     # same shape as sites


# --------------------------------------------------------------------------- #
#                         Full PsiFormer lattice model                        #
# --------------------------------------------------------------------------- #
class LatticePsiFormer(nn.Module):
    n_sites: int
    d_model: int = 256
    depth: int = 6
    n_heads: int = 8
    mlp_dims: Sequence[int] = (512,)

    @nn.compact
    def __call__(self, occ, params, *, train: bool = False):
        """
        occ   : int32[B,N]    lattice occupations (0/1)
        params: float32[B,2]  (t, V)
        """
        B, N = occ.shape
        assert N == self.n_sites, "Mismatch n_sites"

        # Token & positional embeddings -------------------------------------
        tok = nn.Embed(num_embeddings=2, features=self.d_model)(occ)
        pos = self.param('pos_emb',
                         nn.initializers.normal(stddev=0.02),
                         (1, self.n_sites, self.d_model))
        x = tok + pos

        # Stack of residual self-attention blocks ---------------------------
        for _ in range(self.depth):
            x = ResidualSelfAttention(
                    self.d_model, self.n_heads, self.mlp_dims)(x, train=train)

        # Param token & cross-attention -------------------------------------
        p_tok = nn.Dense(self.d_model)(params)[..., None, :]   # (B,1,D)
        x = x + CrossAttentionParams(self.d_model, self.n_heads)(x, p_tok)

        # Pool & output ------------------------------------------------------
        g = x.mean(axis=1)                         # (B, D)
        g = nn.LayerNorm()(g)
        out = nn.Dense(2)(g)                       # (B, 2)  →  Re, Im
        return out
