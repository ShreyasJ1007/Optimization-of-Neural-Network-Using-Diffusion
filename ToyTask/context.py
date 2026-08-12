"""
context.py — enc(B): turning a mini-batch into conditioning for the decoder.

The decoder needs to know WHICH task it is solving. A mini-batch of labelled
examples is the description of that task, so everything here is about squeezing
a variable-length, unordered set of examples down to a fixed set of tokens the
U-Net can attend to.

Order must not matter: shuffling the mini-batch describes the same task, so it
has to produce the same tokens. That is why attention/pooling is used rather
than anything that reads the examples in sequence.
"""

import torch
from einops import rearrange


class MAB(torch.nn.Module):
    """Multihead Attention Block: MAB(X, Y) attends from X to Y."""

    def __init__(self, dim_q, dim_kv, dim_out, heads=4):
        super().__init__()
        self.heads = heads
        self.dim_out = dim_out
        # queries come from X; keys and values come from Y
        self.fc_q = torch.nn.Linear(dim_q, dim_out)
        self.fc_k = torch.nn.Linear(dim_kv, dim_out)
        self.fc_v = torch.nn.Linear(dim_kv, dim_out)
        self.ln0 = torch.nn.LayerNorm(dim_out)
        self.ln1 = torch.nn.LayerNorm(dim_out)
        # the usual transformer feed-forward, widen then narrow
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim_out, dim_out * 2), torch.nn.GELU(),
            torch.nn.Linear(dim_out * 2, dim_out),
        )

    def forward(self, x, y):
        # split the channel dim into heads so each head attends independently
        q = rearrange(self.fc_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.fc_k(y), "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(self.fc_v(y), "b m (h d) -> b h m d", h=self.heads)
        # standard scaled dot-product attention; the sqrt keeps the logits
        # from growing with the head width
        att = torch.softmax(
            q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = rearrange(att @ v, "b h n d -> b n (h d)")   # heads back together
        # residual + norm twice: once around attention, once around the FF.
        # note the residual uses fc_q(x), not x, since x may be a different width.
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
        # lift each raw example (coords + one-hot label) into the working width
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim), torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        )
        # SAB = self-attention block: MAB(z, z). Lets examples compare themselves
        # against each other before anything gets pooled.
        self.sabs = torch.nn.ModuleList(
            [MAB(dim, dim, dim, heads) for _ in range(n_sab)])
        if pooling == "pma":
            # k learned seed vectors; they QUERY the set. Permutation-invariant.
            # small init (0.02) so the seeds start off similar and differentiate
            # during training rather than fighting each other from the start.
            self.seeds = torch.nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
            self.pma = MAB(dim, dim, dim, heads)
        elif pooling != "mean":
            raise ValueError("pooling must be 'pma' or 'mean'")

    def forward(self, batch):
        # tolerate a single un-batched set being passed in
        if batch.dim() == 2:
            batch = batch.unsqueeze(0)
        b = batch.shape[0]
        z = self.embed(batch)
        for sab in self.sabs:
            z = sab(z, z)                       # set elements attend to each other
        if self.pooling == "mean":
            # cheapest possible pooling — invariant, but everything collapses
            # into a single averaged token
            return z.mean(dim=1, keepdim=True)  # DeepSets: (B, 1, dim)
        # PMA: the seeds ask the set questions, so different tokens can pick up
        # different aspects of the task. Still invariant, because attention over
        # the set doesn't care what order the set arrived in.
        seeds = self.seeds.expand(b, -1, -1)    # same seeds for every batch item
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
        # min(8, dim) guards against asking for more groups than channels
        self.norm = torch.nn.GroupNorm(min(8, dim), dim)
        # queries slide along the sequence (Conv1d), keys/values come from the
        # context tokens, which have no sequence axis (Linear)
        self.to_q = torch.nn.Conv1d(dim, dim, 1, bias=False)
        self.to_k = torch.nn.Linear(dim_ctx, dim, bias=False)
        self.to_v = torch.nn.Linear(dim_ctx, dim, bias=False)
        self.to_out = torch.nn.Conv1d(dim, dim, 1)
        # zero-init: at step 0 this returns x untouched, so bolting the block
        # onto a pretrained decoder can't break it
        torch.nn.init.zeros_(self.to_out.weight)
        torch.nn.init.zeros_(self.to_out.bias)

    def forward(self, x, ctx):
        if ctx is None:
            return x                     # unconditioned path, nothing to inject
        res = x                          # keep the input for the residual
        h = self.norm(x)
        q = rearrange(self.to_q(h), "b (nh d) l -> b nh l d", nh=self.heads)
        k = rearrange(self.to_k(ctx), "b m (nh d) -> b nh m d", nh=self.heads)
        v = rearrange(self.to_v(ctx), "b m (nh d) -> b nh m d", nh=self.heads)
        # each of the L positions decides how much of each context token to read
        att = torch.softmax(
            q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = rearrange(att @ v, "b nh l d -> b (nh d) l")
        return res + self.to_out(out)
