"""
paramtokens.py — encoding the MODEL, not just its parameters (Task: arch encoding).

The problem (from supervisor): a flat parameter vector is ambiguous. The same
6 numbers could be "3 Gaussians x 2D mean" or "2 Gaussians x 3D mean"; the same
128 numbers could be a 1-layer CNN, a multi-layer CNN, a Linear layer, or an RNN.
The diffusion model cannot see this structure from the flat vector alone.

The answer here: TOKENIZE BY STRUCTURAL UNIT. Chop the flat vector into its
natural groups (one token per Gaussian component / per neuron / per filter),
project each to a common width, and TAG each with a type/position code that says
what it is. Two consequences:

  * The SAME vector becomes a DIFFERENT set of tokens under a different spec,
    so "3x2D" and "2x3D" are now visibly different inputs.
  * Units that are exchangeable (the Gaussian components; neurons in a layer)
    become tokens that attention treats permutation-invariantly for free.

This is where the architecture is encoded: in (a) how the vector is split into
units, and (b) the per-unit type embedding. Grounded in weight-space /
set-based encoding (Set Transformer; weight-space networks; diffusion-based
weight generation).

A "spec" describes the architecture as a list of unit-groups:
    spec = [UnitGroup(name, count, width), ...]
e.g. 3 Gaussians x 2D:  [UnitGroup("gauss_mean", 3, 2)]
     2 Gaussians x 3D:  [UnitGroup("gauss_mean", 2, 3)]
     tiny MLP:          [UnitGroup("layer0_neuron", 16, 3),
                         UnitGroup("layer1_neuron", 1, 17)]
"""

from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass
class UnitGroup:
    """One group of exchangeable structural units."""
    name: str        # a type label, e.g. "conv1_filter", "gauss_mean"
    count: int       # how many such units
    width: int       # how many scalars per unit


class Spec:
    """An architecture spec: an ordered list of UnitGroups.

    Knows the total parameter count and how to slice a flat vector into units.
    """

    def __init__(self, groups):
        self.groups = groups
        self.total = sum(g.count * g.width for g in groups)
        self.n_units = sum(g.count for g in groups)
        # a distinct integer type-id per group name, for the type embedding
        self.type_ids = []
        for gi, g in enumerate(groups):
            self.type_ids.extend([gi] * g.count)

    def slice(self, flat):
        """(B, total) flat vector -> list of (B, count, width) blocks."""
        out, off = [], 0
        for g in self.groups:
            n = g.count * g.width
            block = flat[:, off:off + n].reshape(flat.shape[0], g.count, g.width)
            out.append(block)
            off += n
        return out


class ParamTokenizer(torch.nn.Module):
    """Flat parameter vector  <->  sequence of unit-tokens.

    forward(flat, spec)  : (B, total) -> (B, n_units, dim)  [tokens]
    untokenize(tok, spec): (B, n_units, dim) -> (B, total)   [back to flat]

    Different-width units are projected to a common `dim`, so architectures
    with different unit widths all become uniform-width token sequences.
    A learned type-embedding tags each token with which group it came from.
    """

    def __init__(self, dim=64, max_unit_width=64, max_types=16):
        super().__init__()
        self.dim = dim
        self.max_unit_width = max_unit_width
        # one shared "in" projection: pad each unit to max_unit_width then project
        self.proj_in = torch.nn.Linear(max_unit_width, dim)
        self.proj_out = torch.nn.Linear(dim, max_unit_width)
        self.type_emb = torch.nn.Embedding(max_types, dim)

    def _pad(self, block):
        # (B, count, width) -> (B, count, max_unit_width)
        b, c, w = block.shape
        if w > self.max_unit_width:
            raise ValueError(f"unit width {w} exceeds max {self.max_unit_width}")
        pad = torch.zeros(b, c, self.max_unit_width - w,
                          dtype=block.dtype, device=block.device)
        return torch.cat([block, pad], dim=-1)

    def forward(self, flat, spec):
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        blocks = spec.slice(flat)                       # per-group units
        tokens = []
        for block in blocks:
            padded = self._pad(block)                   # common width
            tokens.append(self.proj_in(padded))         # -> (B, count, dim)
        tok = torch.cat(tokens, dim=1)                  # (B, n_units, dim)
        # add the type embedding: THIS is the architecture tag per token
        type_ids = torch.tensor(spec.type_ids, device=flat.device)
        tok = tok + self.type_emb(type_ids).unsqueeze(0)
        return tok

    def untokenize(self, tok, spec):
        """Map token predictions back to a flat vector, respecting each unit's
        true width (padding is discarded)."""
        raw = self.proj_out(tok)                        # (B, n_units, max_width)
        out, ui = [], 0
        for g in spec.groups:
            block = raw[:, ui:ui + g.count, :g.width]   # trim to real width
            out.append(block.reshape(tok.shape[0], g.count * g.width))
            ui += g.count
        return torch.cat(out, dim=1)                    # (B, total)
