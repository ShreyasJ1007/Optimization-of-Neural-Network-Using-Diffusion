"""
tokenopt.py — a score network over STRUCTURAL UNIT TOKENS, not a flat sequence.

WHY THIS REPLACES THE U-NET
    The 1D U-Net treated the parameter vector as a signal and downsampled it
    (12 -> 6 -> 3). But the vector is

        [ mu_1x, mu_1y, logs_1x, logs_1y | mu_2x, ... | mu_3x, ... ]

    so stride-2 convolution blends a component's mean with its own log-std, then
    blends across components entirely - destroying exactly the structure that
    matters. Convolution also assumes locality and translation-invariance, and
    neither holds for a parameter vector: positions 0 and 4 are semantically
    parallel but four apart.

    Here each STRUCTURAL UNIT is one token (one Gaussian component, one neuron,
    one filter), and self-attention relates them. Three consequences:

      * structure preserved  - a unit is never blended with its neighbours
      * permutation-equivariant - swapping two components swaps two tokens, and
        the output permutes to match, which is the correct symmetry
      * variable size native - K=2 is simply 2 tokens instead of 3, so the
        2 -> 3 Gaussian generalisation test becomes possible at all
        (the U-Net was hard-wired to a fixed vector length)

CONDITIONING
    Cross-attention to the mini-batch context at EVERY block - Task-5 option 4,
    following Latent Diffusion. The earlier input-level injection closed 1.4% of
    the optimal-vs-random gap; per-block closed 52%.
"""

import math

import torch

from paramtokens import ParamTokenizer, Spec, UnitGroup
from context import SetEncoder
from gradtts1d import get_noise


class SinusoidalPosEmb(torch.nn.Module):
    """Timestep scalar -> a rich vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Attention(torch.nn.Module):
    """Multihead attention from `x` to `y`, with no internal norm or residual."""

    def __init__(self, dim_q, dim_kv, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.to_out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, x, y):
        b, n, d = x.shape
        h = self.heads
        q = self.to_q(x).reshape(b, n, h, d // h).transpose(1, 2)
        k = self.to_k(y).reshape(b, y.shape[1], h, d // h).transpose(1, 2)
        v = self.to_v(y).reshape(b, y.shape[1], h, d // h).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / ((d // h) ** 0.5), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)


class TokenBlock(torch.nn.Module):
    """Pre-norm transformer block: self-attention over units, cross-attention
    to the context, then a feed-forward layer.

    CLEAN RESIDUALS MATTER HERE. An earlier version reused the MAB block from
    context.py, whose residual is LayerNorm(proj(x) + attn) - the input passes
    through a learned projection and is re-normalised at every block. Across
    four blocks that repeatedly discards magnitude, and noise prediction depends
    on magnitude. Training loss stayed low while sampling degraded. Pre-norm
    with an untouched residual stream preserves scale.

    Self-attention is permutation-equivariant, so exchangeable units behave
    correctly.
    """

    def __init__(self, dim, ctx_dim, heads=4):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim)
        self.self_attn = Attention(dim, dim, heads)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, ctx_dim, heads)
        self.ln3 = torch.nn.LayerNorm(dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, tok, ctx):
        tok = tok + self.self_attn(self.ln1(tok), self.ln1(tok))
        if ctx is not None:
            tok = tok + self.cross_attn(self.ln2(tok), ctx)
        tok = tok + self.ff(self.ln3(tok))
        return tok


class TokenScoreNet(torch.nn.Module):
    """s_theta(w, spec, t, enc(B)) -> predicted noise, same shape as w."""

    def __init__(self, dim=128, ctx_dim=64, heads=4, n_layers=4,
                 max_unit_width=8):
        super().__init__()
        self.tokenizer = ParamTokenizer(dim=dim, max_unit_width=max_unit_width)
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )
        self.blocks = torch.nn.ModuleList(
            [TokenBlock(dim, ctx_dim, heads) for _ in range(n_layers)])
        self.out_norm = torch.nn.LayerNorm(dim)  # final norm only

    def forward(self, w, spec, t, ctx=None):
        tok = self.tokenizer(w, spec)                     # (B, n_units, dim)
        temb = self.time_mlp(self.time_pos_emb(t))        # (B, dim)
        tok = tok + temb.unsqueeze(1)                     # every token sees t
        for blk in self.blocks:
            tok = blk(tok, ctx)
        return self.tokenizer.untokenize(self.out_norm(tok), spec)


class TokenDiffusionOptimiser(torch.nn.Module):
    """Algorithm 1 (training) and Algorithm 2 (sampling), over unit tokens.

    Unlike the U-Net version this is NOT tied to a fixed parameter count - the
    spec is passed per call, so one trained model can be asked for K=2 or K=3.
    """

    def __init__(self, feat_dim, dim=128, ctx_dim=64, n_ctx=8, heads=4,
                 n_layers=4, n_timesteps=50, beta_min=0.05, beta_max=20.0,
                 max_unit_width=8):
        super().__init__()
        self.score_net = TokenScoreNet(dim, ctx_dim, heads, n_layers,
                                       max_unit_width)
        self.set_encoder = SetEncoder(in_dim=feat_dim, dim=ctx_dim,
                                      n_tokens=n_ctx)
        self.beta_min, self.beta_max = beta_min, beta_max
        self.n_timesteps = n_timesteps
        self.norm_stats = {}

    # -- normalisation, PER UNIT-POSITION so it generalises across K ---------
    def fit_normaliser(self, w_pool, spec):
        """Statistics per (group, position-within-unit).

        Because every unit of a group shares a layout, these statistics do not
        depend on how MANY units there are - so a normaliser fitted on K=3
        transfers unchanged to K=2.
        """
        for g, block in zip(spec.groups, spec.slice(w_pool)):
            flat = block.reshape(-1, g.width)
            self.norm_stats[g.name] = (flat.mean(0),
                                       flat.std(0).clamp_min(1e-3))

    def normalise(self, w, spec):
        out = []
        for g, block in zip(spec.groups, spec.slice(w)):
            m, s = self.norm_stats[g.name]
            out.append(((block - m) / s).reshape(w.shape[0], -1))
        return torch.cat(out, dim=1)

    def denormalise(self, w, spec):
        out = []
        for g, block in zip(spec.groups, spec.slice(w)):
            m, s = self.norm_stats[g.name]
            out.append((block * s + m).reshape(w.shape[0], -1))
        return torch.cat(out, dim=1)

    # -- ALGORITHM 1: training ---------------------------------------------
    def compute_loss(self, w0, batch_feats, spec, offset=1e-5):
        b = w0.shape[0]
        mu = torch.zeros_like(w0)                       # normalised => mu = 0
        t = torch.rand(b, device=w0.device).clamp(offset, 1.0 - offset)

        cum = get_noise(t.unsqueeze(-1), self.beta_min, self.beta_max,
                        cumulative=True)
        mean = w0 * torch.exp(-0.5 * cum) + mu * (1.0 - torch.exp(-0.5 * cum))
        var = 1.0 - torch.exp(-cum)
        z = torch.randn_like(w0)
        wt = mean + z * torch.sqrt(var)

        ctx = self.set_encoder(batch_feats)
        pred = self.score_net(wt, spec, t, ctx) * torch.sqrt(var)
        return ((pred + z) ** 2).mean()

    # -- ALGORITHM 2: sampling ---------------------------------------------
    @torch.no_grad()
    def sample(self, batch_feats, spec, n_timesteps=None, stoc=False,
               generator=None):
        n_timesteps = n_timesteps or self.n_timesteps
        if batch_feats.dim() == 2:
            batch_feats = batch_feats.unsqueeze(0)
        b = batch_feats.shape[0]
        device = batch_feats.device

        ctx = self.set_encoder(batch_feats)
        mu = torch.zeros(b, spec.total, device=device)
        w = mu + torch.randn(b, spec.total, device=device, generator=generator)

        h = 1.0 / n_timesteps
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(b, device=device)
            beta_t = get_noise(t.unsqueeze(-1), self.beta_min, self.beta_max,
                               cumulative=False)
            score = self.score_net(w, spec, t, ctx)
            if stoc:
                dxt = (0.5 * (mu - w) - score) * beta_t * h
                dxt = dxt + torch.randn_like(w) * torch.sqrt(beta_t * h)
            else:
                dxt = 0.5 * (mu - w - score) * beta_t * h
            w = w - dxt

        return self.denormalise(w, spec)
