from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

class TransformerWavefunction(nn.Module):
    input_dim: int
    model_dim: int
    num_heads: int
    num_layers: int
    output_dim: int
    max_seq_len: int = 100  # Assuming max sequence length of 100

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.input_dim, features=self.model_dim)
        self.positional_encoding = self.param(
            "positional_encoding", 
            lambda rng, shape: jnp.zeros(shape), 
            (1, self.max_seq_len, self.model_dim)
        )
        self.transformer_layers = [
            nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim,
                use_bias=True
            ) for _ in range(self.num_layers)
        ]
        self.fc = nn.Dense(self.output_dim)

    def __call__(self, configuration):
        # Embed the input configuration
        x = self.embedding(configuration) + self.positional_encoding[:, :configuration.shape[1], :]

        # Pass through the transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Map to wavefunction output
        wavefunction = self.fc(x)
        return wavefunction

# Example usage
if __name__ == "__main__":
    input_dim = 10  # Number of discrete states
    model_dim = 64  # Transformer model dimension
    num_heads = 4   # Number of attention heads
    num_layers = 2  # Number of transformer layers
    output_dim = 1  # Wavefunction output dimension

    model = TransformerWavefunction(input_dim, model_dim, num_heads, num_layers, output_dim)
    rng = PRNGKey(0)
    configuration = jax.random.randint(rng, (8, 10), 0, input_dim)  # Batch of 8 configurations, each of length 10
    params = model.init(rng, configuration)  # Initialize parameters
    wavefunction = model.apply(params, configuration)
    print(wavefunction.shape)  # Should output (8, 10, 1)