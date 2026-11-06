"""Hyperbolic layers used by the StackRec heads.

Two models of hyperbolic space are implemented:

* a Poincare-ball next-item head (``HyperbolicHead``), scoring items by
  negative distance in the ball of curvature ``-c``;
* a Lorentz/hyperboloid residual head (``LorentzResidualHead``) that both
  scores items in the Lorentz model and returns a curvature-dependent
  residual vector for the hidden state of a copied StackRec group.

The Lorentz residual is the difference between the spatial part of the
exponential map onto the hyperboloid and the original tangent vector.  It is
exactly zero at ``c = 0`` (Euclidean limit), so a curvature ramp from zero
introduces the hyperbolic geometry continuously.  ``run_stackrec.py`` uses
this combined with a growing residual gate and a norm-preserving addition.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

_EPS = 1.0e-7


def _to_tensor(value, reference):
    if torch.is_tensor(value):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.tensor(float(value), dtype=reference.dtype, device=reference.device)


def expmap0(x, curvature):
    """Exponential map from the tangent space at the origin onto the ball."""
    c = _to_tensor(curvature, x)
    sqrt_c = torch.sqrt(c.clamp_min(_EPS))
    norm = x.norm(dim=-1, keepdim=True)
    coeff = torch.tanh(sqrt_c * norm) / (sqrt_c * norm).clamp_min(_EPS)
    return coeff * x


def poincare_distance(u, v, curvature):
    """Distance in the Poincare ball.

    ``u`` has shape ``(..., d)`` and ``v`` has shape ``(n, d)``.  The result has
    shape ``(..., n)``.  The computation uses
    ``||u - v||^2 = ||u||^2 + ||v||^2 - 2 <u, v>`` so it never materialises the
    full pairwise difference tensor.
    """
    c = _to_tensor(curvature, u)
    sqrt_c = torch.sqrt(c.clamp_min(_EPS))

    u_sq = (u * u).sum(dim=-1, keepdim=True)  # (..., 1)
    v_sq = (v * v).sum(dim=-1)  # (n)
    dot = u @ v.transpose(-1, -2)  # (..., n)
    diff_sq = (u_sq + v_sq.unsqueeze(-2) - 2.0 * dot).clamp_min(0.0)

    denom = (1.0 - c * u_sq).clamp_min(_EPS) * (
        1.0 - c * v_sq.unsqueeze(-2)
    ).clamp_min(_EPS)
    arg = (1.0 + 2.0 * c * diff_sq / denom).clamp_min(1.0 + _EPS)
    return (2.0 / sqrt_c) * torch.acosh(arg)


def _sinhc(value):
    """``sinh(value) / value`` with the removable singularity at zero.

    ``value`` is expected to be non-negative.  The series expansion is used
    for small arguments so that the function and its gradient are finite when
    the curvature is exactly zero.
    """
    return torch.where(
        value < 1.0e-4,
        1.0 + value * value / 6.0,
        torch.sinh(value) / value.clamp_min(_EPS),
    )


def lorentz_expmap0(x, curvature, max_argument=20.0):
    """Exponential map at the origin onto the Lorentz hyperboloid.

    Parameters
    ----------
    x:
        Tangent vectors at the origin, shape ``(..., d)``.
    curvature:
        Magnitude ``c`` of the negative curvature.  ``c = 0`` is allowed and
        gives the Euclidean limit (the returned time coordinate is then only
        a numerical artefact and should not be used).

    Returns
    -------
    (time, spatial):
        ``time`` has shape ``(..., 1)``, ``spatial`` has shape ``(..., d)``.
        They satisfy ``-time**2 + ||spatial||**2 = -1 / c`` for ``c > 0``.
    """
    c = _to_tensor(curvature, x)
    sqrt_c = torch.sqrt(c.clamp_min(_EPS))
    norm = x.norm(dim=-1, keepdim=True)
    # a = sqrt(c) * ||x||; clamping keeps cosh/sinh finite for extreme states.
    a = (sqrt_c * norm).clamp(max=float(max_argument))
    sinc = _sinhc(a)
    spatial = sinc * x
    time = torch.cosh(a) / sqrt_c
    return time, spatial


def lorentz_distance(u_time, u_spatial, v_time, v_spatial, curvature):
    """Distance in the Lorentz hyperboloid model.

    ``u_*`` has shape ``(..., 1)`` / ``(..., d)`` and ``v_*`` has shape
    ``(n, 1)`` / ``(n, d)``.  The result has shape ``(..., n)``.  The
    Euclidean limit ``||u_spatial - v_spatial||`` is used for ``c`` close to
    zero to avoid the ``0 / 0`` indeterminacy of ``acosh(1 + O(c)) / sqrt(c)``.
    """
    if u_spatial.shape[-1] != v_spatial.shape[-1]:
        raise ValueError("Lorentz distance requires matching spatial widths")
    c = _to_tensor(curvature, u_spatial)
    sqrt_c = torch.sqrt(c.clamp_min(_EPS))

    # Stable form of ``-c <u, v>_L``:
    #   -c <u, v>_L = 1 + (c / 2) * <u - v, u - v>_L
    # ``cdist`` computes the pairwise Euclidean part without materialising
    # the (num_events, num_classes, d) difference tensor.
    spatial_diff = torch.cdist(u_spatial, v_spatial, p=2)
    time_diff = u_time - v_time.transpose(-1, -2)
    minkowski_diff_sq = spatial_diff.pow(2) - time_diff.pow(2)
    arg = (1.0 + 0.5 * c * minkowski_diff_sq).clamp_min(1.0)
    hyperbolic = torch.acosh(arg) / sqrt_c

    return torch.where(c > 1.0e-4, hyperbolic, spatial_diff)


def _inverse_softplus(value, eps=1.0e-6):
    value = max(float(value), eps)
    return math.log(math.expm1(value))


class HyperbolicHead(nn.Module):
    """Hyperbolic next-item head at a StackRec group boundary.

    Parameters
    ----------
    embedding_dim:
        Width of the transformer hidden state.
    num_classes:
        Number of item ids (including padding / mask ids).
    curvature:
        Initial / fixed curvature magnitude ``c`` of the ball.
    learnable_curvature:
        If true, ``c`` is learned (positive through softplus).
    distance:
        ``"squared"`` (default) or ``"plain"`` distance scoring.
    temperature:
        Initial scale of the distance logits (learned).
    """

    def __init__(
        self,
        embedding_dim,
        num_classes,
        curvature=1.0,
        learnable_curvature=False,
        distance="squared",
        temperature=0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.distance_type = str(distance).lower()
        if self.distance_type not in ("squared", "plain"):
            raise ValueError(f"Unknown distance type: {distance}")

        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.item_proj = nn.Linear(embedding_dim, embedding_dim)
        for layer in (self.proj, self.item_proj):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)

        self.learnable_curvature = bool(learnable_curvature)
        if self.learnable_curvature:
            self.raw_curvature = nn.Parameter(
                torch.tensor(_inverse_softplus(curvature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "curvature_value",
                torch.tensor(float(curvature), dtype=torch.float32),
                persistent=True,
            )

        self.logit_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(temperature), dtype=torch.float32)
        )

    def curvature(self):
        if self.learnable_curvature:
            return F.softplus(self.raw_curvature) + 1.0e-6
        return self.curvature_value

    def logit_scale(self):
        return F.softplus(self.logit_scale_raw) + 1.0e-6

    def forward(self, hidden_states, item_weights):
        """Return hyperbolic item logits.

        ``hidden_states`` is ``(num_events, d)`` and ``item_weights`` is
        ``(num_classes, d)``.
        """
        c = self.curvature()
        z = expmap0(self.proj(hidden_states), c)
        v = expmap0(self.item_proj(item_weights), c)
        distance = poincare_distance(z, v, c)
        if self.distance_type == "squared":
            distance = distance.pow(2)
        return -self.logit_scale() * distance


class LorentzResidualHead(nn.Module):
    """Lorentz/hyperboloid head with a curvature-gated residual output.

    The head plays two roles at a StackRec group boundary:

    1. it scores items by (negative) squared or plain distance in the Lorentz
       hyperboloid of curvature ``-c`` (the usual next-item cross-entropy can
       be applied to these logits);
    2. it returns a Euclidean residual
       ``r = (sinhc(sqrt(c) * ||u||) - 1) * W_out u`` with ``u = W_in h``.
       This is the difference between the spatial part of the Lorentz
       exponential map and its tangent vector ``u``, i.e. the pure
       curvature-induced correction.  It is **exactly zero** for ``c = 0``.

    The residual is not applied here: ``SequentialEncoder`` gates it, matches
    its norm to the boundary state and renormalises the sum, so the norm of
    the representation entering the pretrained subnet is preserved.

    Parameters
    ----------
    embedding_dim:
        Width of the transformer hidden state.
    num_classes:
        Number of item ids (including padding / mask ids).
    curvature:
        Initial curvature magnitude ``c`` of the hyperboloid.  ``0`` is the
        Euclidean limit and gives a zero residual.
    learnable_curvature:
        If true, ``c`` is learned (positive through softplus).  The scheduled
        value is then used as initialisation and can still be overwritten by
        the curvature ramp through :meth:`set_curvature`.
    distance:
        ``"squared"`` (default) or ``"plain"`` distance scoring.
    temperature:
        Initial scale of the distance logits (learned).
    """

    def __init__(
        self,
        embedding_dim,
        num_classes,
        curvature=0.0,
        learnable_curvature=False,
        distance="squared",
        temperature=0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.distance_type = str(distance).lower()
        if self.distance_type not in ("squared", "plain"):
            raise ValueError(f"Unknown distance type: {distance}")

        # Item-scoring branch (same shape as HyperbolicHead).
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.item_proj = nn.Linear(embedding_dim, embedding_dim)
        # Residual branch.  A separate projection keeps the auxiliary CE from
        # forcing the residual direction, and vice versa.
        self.res_proj = nn.Linear(embedding_dim, embedding_dim)
        self.res_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        for layer in (self.proj, self.item_proj, self.res_proj, self.res_out):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.learnable_curvature = bool(learnable_curvature)
        if self.learnable_curvature:
            self.raw_curvature = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(max(float(curvature), 0.0)),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "curvature_value",
                torch.tensor(float(curvature), dtype=torch.float32),
                persistent=True,
            )

        self.logit_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(temperature), dtype=torch.float32)
        )

    def curvature(self):
        if self.learnable_curvature:
            return F.softplus(self.raw_curvature) + 1.0e-6
        return self.curvature_value

    def set_curvature(self, value):
        """Overwrite the curvature magnitude (used by the stage ramp)."""
        value = max(float(value), 0.0)
        with torch.no_grad():
            if self.learnable_curvature:
                self.raw_curvature.fill_(_inverse_softplus(value))
            else:
                self.curvature_value.fill_(value)

    def logit_scale(self):
        return F.softplus(self.logit_scale_raw) + 1.0e-6

    def residual(self, hidden_states):
        """Return the raw curvature-induced residual ``(num_events, d)``."""
        c = self.curvature().clamp_min(0.0)
        # c is a non-learned buffer during the ramp, so sqrt(0) is safe here;
        # using the exact zero keeps the residual bitwise zero at c = 0.
        sqrt_c = torch.sqrt(c)
        u = self.res_proj(hidden_states)
        a = (sqrt_c * u.norm(dim=-1, keepdim=True)).clamp(max=20.0)
        correction = _sinhc(a) - 1.0
        return correction * self.res_out(u)

    def forward(self, hidden_states, item_weights):
        """Return Lorentz item logits with shape ``(num_events, num_classes)``."""
        c = self.curvature()
        u = self.proj(hidden_states)
        v = self.item_proj(item_weights)
        u_time, u_spatial = lorentz_expmap0(u, c)
        v_time, v_spatial = lorentz_expmap0(v, c)
        distance = lorentz_distance(u_time, u_spatial, v_time, v_spatial, c)
        if self.distance_type == "squared":
            distance = distance.pow(2)
        return -self.logit_scale() * distance
