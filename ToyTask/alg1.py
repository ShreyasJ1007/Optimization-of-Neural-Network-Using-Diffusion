"""
alg1.py — Algorithm 1 (training) and Algorithm 2 (sampling), wired to the toy task.

WHAT THIS ADDS OVER pipeline.py
  pipeline.py only ran the reverse process, and injected context once before the
  loop. Training needs two more things:
    - the FORWARD path: corrupt a clean w_0 and ask the network to name the noise
    - context available at EVERY score evaluation, not once up front
  Both are handled here by routing every call through `_score`.

THE SCORE NETWORK
  s_theta(w_t, mu, t, enc(B)) is the Grad-TTS 1D U-Net estimator, with the
  mini-batch context cross-attended into the weight sequence before each call.

NORMALISATION
  Raw parameters are on mixed scales (means span several units, log-stds sit
  near zero). Diffusion assumes roughly unit scale, so parameters are
  standardised using pool statistics. In normalised space the reference mean mu
  is simply 0, which is what the forward process diffuses toward.
"""

import torch

from gradtts1d import Diffusion, GradLogPEstimator2d, Mish, get_noise
from context import SetEncoder, CrossAttention


class ConditionedEstimator(GradLogPEstimator2d):
    """
    The 1D U-Net, with context injected at EVERY block.
    The forward body is the parent's, with one added line marked below.
    """

    def __init__(self, dim, ctx_dim, **kw):
        super().__init__(dim, **kw)
        self.ctx_mlp = torch.nn.Sequential(
            torch.nn.Linear(ctx_dim, dim * 4), Mish(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x, mask, mu, t, ctx_vec=None):
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        # >>> the one change: context joins the per-block conditioning signal
        if ctx_vec is not None:
            t = t + self.ctx_mlp(ctx_vec)
        # <<<

        x = torch.stack([mu, x], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)
        return (output * mask).squeeze(1)


class DiffusionOptimiser(torch.nn.Module):
    """A parametric optimiser: generates network parameters from noise.

    Args:
        n_params:  length of the parameter vector to generate.
        feat_dim:  dimensionality of one mini-batch element (coords + one-hot).
        dim:       U-Net base width.
        ctx_dim:   context-token width.
        n_ctx:     number of context tokens enc(B) emits.
    """

    def __init__(self, n_params, feat_dim, dim=64, ctx_dim=64, n_ctx=8,
                 n_timesteps=50, beta_min=0.05, beta_max=20.0,
                 cond_mode="perblock"):
        super().__init__()
        self.n_params = n_params
        self.n_timesteps = n_timesteps
        self.cond_mode = cond_mode      # 'perblock' (option 3/4) or 'input' (option 1)

        # the diffusion process + its U-Net score estimator
        self.diffusion = Diffusion(n_feats=1, dim=dim,
                                   beta_min=beta_min, beta_max=beta_max)
        if cond_mode == "perblock":
            # swap in the estimator that conditions every block
            self.diffusion.estimator = ConditionedEstimator(
                dim, ctx_dim=ctx_dim, n_feats=1)

        # enc(B): labelled mini-batch -> permutation-invariant context tokens
        self.set_encoder = SetEncoder(in_dim=feat_dim, dim=ctx_dim, n_tokens=n_ctx)

        # context injection into the weight sequence (zero-init => no-op at start)
        self.lift = torch.nn.Conv1d(1, dim, 1)
        self.inject = CrossAttention(dim=dim, dim_ctx=ctx_dim)
        self.project = torch.nn.Conv1d(dim, 1, 1)
        torch.nn.init.zeros_(self.project.weight)
        torch.nn.init.zeros_(self.project.bias)

        # normalisation buffers, filled by fit_normaliser()
        self.register_buffer("w_mean", torch.zeros(n_params))
        self.register_buffer("w_std", torch.ones(n_params))

    # -- normalisation 
    def fit_normaliser(self, w_pool):
        """Standardise using pool statistics. w_pool: (n_tasks, n_params)."""
        self.w_mean.copy_(w_pool.mean(dim=0))
        self.w_std.copy_(w_pool.std(dim=0).clamp_min(1e-3))

    def normalise(self, w):
        return (w - self.w_mean) / self.w_std

    def denormalise(self, w):
        return w * self.w_std + self.w_mean

    # -- the conditioned score 
    def encode_context(self, batch_feats):
        """batch_feats: (B, N, feat_dim) -> context tokens (B, n_ctx, ctx_dim)."""
        return self.set_encoder(batch_feats)

    def _apply_context(self, w, ctx):
        """Fold context into the weight sequence. (B, L) -> (B, L)."""
        h = self.lift(w.unsqueeze(1))
        h = self.inject(h, ctx)
        return w + self.project(h).squeeze(1)

    def _score(self, w, mask, mu, t, ctx):
        """s_theta(w, mu, t, enc(B)) — one conditioned score evaluation."""
        if self.cond_mode == "perblock":
            # pool the context tokens to one vector, then condition every block
            ctx_vec = ctx.mean(dim=1) if ctx is not None else None
            return self.diffusion.estimator(w, mask, mu, t, ctx_vec=ctx_vec)
        # 'input' mode: fold context into the sequence once, before the U-Net
        w_cond = self._apply_context(w, ctx) if ctx is not None else w
        return self.diffusion.estimator(w_cond, mask, mu, t)

    # -- ALGORITHM 1: training 
    def compute_loss(self, w0, batch_feats, offset=1e-5):
        """One training step of Algorithm 1.

        w0:          (B, n_params) clean target parameters, already normalised.
        batch_feats: (B, N, feat_dim) the labelled mini-batches.
        Returns the noise-prediction loss.
        """
        b = w0.shape[0]
        device = w0.device
        mask = torch.ones_like(w0)
        mu = torch.zeros_like(w0)                  # normalised space => mu = 0

        # step 2: sample a noise level
        t = torch.rand(b, dtype=w0.dtype, device=device)
        t = torch.clamp(t, offset, 1.0 - offset)

        # steps 3-4: forward-diffuse toward mu, keeping the injected noise z
        xt, z = self.diffusion.forward_diffusion(w0, mask, mu, t)

        # step 5: predict the noise, conditioned on the mini-batch
        ctx = self.encode_context(batch_feats)
        cum_noise = get_noise(t.unsqueeze(-1), self.diffusion.beta_min,
                            self.diffusion.beta_max, cumulative=True)
        pred = self._score(xt, mask, mu, t, ctx)
        pred = pred * torch.sqrt(1.0 - torch.exp(-cum_noise))

        # step 6: squared error to the true noise
        return ((pred + z) ** 2).sum() / (mask.sum())

    #ALGORITHM 2: sampling
    @torch.no_grad()
    
    def sample(self, batch_feats, n_timesteps=None, stoc=False, generator=None):
        """
        Generate parameters for the task described by batch_feats.
        Returns DENORMALISED parameters, ready to use.
        """
        n_timesteps = n_timesteps or self.n_timesteps
        if batch_feats.dim() == 2:
            batch_feats = batch_feats.unsqueeze(0)
        b = batch_feats.shape[0]
        device = self.w_mean.device

        mask = torch.ones(b, self.n_params, device=device)
        mu = torch.zeros(b, self.n_params, device=device)
        ctx = self.encode_context(batch_feats)

        # step 1: start from the prior plus noise
        w = mu + torch.randn(b, self.n_params, device=device, generator=generator)

        # step 2: walk time backwards
        h = 1.0 / n_timesteps
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(b, device=device)
            beta_t = get_noise(t.unsqueeze(-1), self.diffusion.beta_min,
                               self.diffusion.beta_max, cumulative=False)
            score = self._score(w, mask, mu, t, ctx)
            if stoc:
                dxt = (0.5 * (mu - w) - score) * beta_t * h
                dxt = dxt + torch.randn_like(w) * torch.sqrt(beta_t * h)
            else:
                dxt = 0.5 * (mu - w - score) * beta_t * h
            w = w - dxt

        # step 3: return, denormalised
        return self.denormalise(w)
