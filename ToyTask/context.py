"""
context.py — enc(B): turning a mini-batch into conditioning for the decoder.

TASK 5/6 IMPLEMENTATION. Two modules, both literature-grounded:

SetEncoder      the "how" - compress a variable-size, UNORDERED mini-batch
                into a fixed number of context tokens.
                Follows Set Transformer (Lee et al., ICML 2019): per-element
                MLP -> self-attention over the set (SAB) -> Pooling by
                Multihead Attention (PMA) with k learned seed vectors.
                Permutation-invariant by construction: shuffling the batch
                does not change the output. Mean-pooling (DeepSets, Zaheer
                et al. 2017) is available as a baseline for ablation.

CrossAttention  the "where" - inject those tokens into the U-Net.
                Follows Latent Diffusion (Rombach et al., CVPR 2022):
                queries come from the U-Net feature map, keys/values from
                the context tokens, applied at each attention site.

Why tokens rather than one vector: FiD (Izacard & Grave, EACL 2021) encodes
each passage independently and lets the decoder attend over the concatenation,
which avoids collapsing everything into a single vector too early.

"""

import torch
from einops import rearrange


class MAB(torch.nn.Module):
    """Multihead Attention Block: MAB(X, Y) attends from X to Y."""

    def __init__(self, dim_q, dim_kv, dim_out, heads=4):
        super().__init__()
        self.heads = heads
        self.dim_out = dim_out
        self.fc_q = torch.nn.Linear(dim_q, dim_out)
        self.fc_k = torch.nn.Linear(dim_kv, dim_out)
        self.fc_v = torch.nn.Linear(dim_kv, dim_out)
        self.ln0 = torch.nn.LayerNorm(dim_out)
        self.ln1 = torch.nn.LayerNorm(dim_out)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim_out, dim_out * 2), torch.nn.GELU(),
            torch.nn.Linear(dim_out * 2, dim_out),
        )

    def forward(self, x, y):
        q = rearrange(self.fc_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.fc_k(y), "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(self.fc_v(y), "b m (h d) -> b h m d", h=self.heads)
        att = torch.softmax(
            q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = rearrange(att @ v, "b h n d -> b n (h d)")
        h = self.ln0(self.fc_q(x) + out)
        return self.ln1(h + self.ff(h))


class SetEncoder(torch.nn.Module):
    """enc(B): mini-batch -> fixed number of context tokens.

    Args:
        in_dim:      dimensionality of one example in the batch.
        dim:         internal / output token width.
        n_tokens:    k, how many context tokens to emit (PMA seeds).
        n_sab:       how many self-attention blocks over the set.
        pooling:     'pma' (Set Transformer) or 'mean' (DeepSets baseline).

    Input : (B, N, in_dim)  - N examples, order irrelevant. Also accepts
                              (N, in_dim) and adds the batch dim.
    Output: (B, n_tokens, dim)
    """

    def __init__(self, in_dim, dim=128, n_tokens=8, n_sab=2, heads=4,
                 pooling="pma"):
        super().__init__()
        self.pooling = pooling
        self.n_tokens = n_tokens
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim), torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        )
        self.sabs = torch.nn.ModuleList(
            [MAB(dim, dim, dim, heads) for _ in range(n_sab)])
        if pooling == "pma":
            # k learned seed vectors; they QUERY the set. Permutation-invariant.
            self.seeds = torch.nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
            self.pma = MAB(dim, dim, dim, heads)
        elif pooling != "mean":
            raise ValueError("pooling must be 'pma' or 'mean'")

    def forward(self, batch):
        if batch.dim() == 2:
            batch = batch.unsqueeze(0)
        b = batch.shape[0]
        z = self.embed(batch)
        for sab in self.sabs:
            z = sab(z, z)                       # set elements attend to each other
        if self.pooling == "mean":
            return z.mean(dim=1, keepdim=True)  # DeepSets: (B, 1, dim)
        seeds = self.seeds.expand(b, -1, -1)
        return self.pma(seeds, z)               # PMA: (B, n_tokens, dim)


class CrossAttention(torch.nn.Module):
    """Inject context tokens into a U-Net feature map (Rombach et al. 2022).

    Queries from the features, keys/values from the context. Zero-initialised
    output projection, so at init the block is a no-op and cannot destabilise
    a working decoder (same idea as Grad-TTS's Rezero).

    Input : x (B, C, L) feature map, ctx (B, k, dim_ctx)
    Output: (B, C, L)
    """

    def __init__(self, dim, dim_ctx, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = torch.nn.GroupNorm(min(8, dim), dim)
        self.to_q = torch.nn.Conv1d(dim, dim, 1, bias=False)
        self.to_k = torch.nn.Linear(dim_ctx, dim, bias=False)
        self.to_v = torch.nn.Linear(dim_ctx, dim, bias=False)
        self.to_out = torch.nn.Conv1d(dim, dim, 1)
        torch.nn.init.zeros_(self.to_out.weight)
        torch.nn.init.zeros_(self.to_out.bias)

    def forward(self, x, ctx):
        if ctx is None:
            return x
        res = x
        h = self.norm(x)
        q = rearrange(self.to_q(h), "b (nh d) l -> b nh l d", nh=self.heads)
        k = rearrange(self.to_k(ctx), "b m (nh d) -> b nh m d", nh=self.heads)
        v = rearrange(self.to_v(ctx), "b m (nh d) -> b nh m d", nh=self.heads)
        att = torch.softmax(
            q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = rearrange(att @ v, "b nh l d -> b (nh d) l")
        return res + self.to_out(out)
