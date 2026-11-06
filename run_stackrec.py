"""AdamW + StackRec for SASRec.

The experiment starts from a 6-layer SASRec and progressively activates its
layers:

    last 2 layers  ->  middle 4 layers  ->  all 6 layers

New layers are initialised from already trained layers and are introduced
through a per-layer forward coefficient that is ramped from 0 to 1.  The whole
procedure is run as one training process; validation/test metrics are recorded
after every stage.

The training script is written to work with three encoder variants:
  * the original encoder that exposes both Poincare hyperbolic heads and
    Lorentz residual heads;
  * a purely Euclidean encoder that exposes neither (all hyperbolic blocks are
    silently disabled if requested);
  * a Lorentz-only encoder that exposes only ``add_lorentz_head`` /
    ``set_lorentz_heads_trainable`` / ``set_lorentz_gate`` /
    ``get_lorentz_gates`` / ``set_curvature``.
"""

import copy
import inspect
import json
import time
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from run_model import build_metrics, build_model, prepare_data
from src.optimizer import BasicOptimizer, Muon, MuonAdamW
from src.utils import create_logger, fix_random_seed, inference, move_batch

logger = create_logger(name=__name__)


KEY_METRICS = [
    "ndcg@10",
    "recall@10",
    "ndcg@100",
    "recall@100",
    "coverage@10",
    "coverage@100",
    "meanrank@10",
    "meanrank@100",
]


def select_key_metrics(metrics):
    return {name: metrics[name] for name in KEY_METRICS if name in metrics}


# ---------------------------------------------------------------------------
# Capability / compatibility helpers
# ---------------------------------------------------------------------------
# These make the training loop work with any of the three encoder variants:
#   * the original encoder (both heads);
#   * the Euclidean-only encoder (no heads at all);
#   * the Lorentz-only encoder (only ``add_lorentz_head`` etc.).
# Every "advanced" call goes through a helper that falls back gracefully.
# ---------------------------------------------------------------------------


def _has_method(obj, name):
    return callable(getattr(obj, name, None))


def _filter_kwargs(func, **kwargs):
    """Keep only kwargs accepted by ``func``'s signature.

    If the callable accepts ``**kwargs`` it is returned unchanged.  This lets
    us pass ``temperature`` / ``distance`` to whichever encoder variant is
    installed, even if a variant's ``add_lorentz_head`` does not accept them.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# --- hyperbolic (Poincare) heads -------------------------------------------


def _supports_hyperbolic_heads(model):
    return _has_method(model, "add_hyperbolic_head")


def _set_hyperbolic_heads_trainable(model, layer_indices, trainable):
    if _has_method(model, "set_hyperbolic_heads_trainable"):
        model.set_hyperbolic_heads_trainable(layer_indices, trainable)


def _get_hyperbolic_curvatures(model):
    if _has_method(model, "get_hyperbolic_curvatures"):
        return model.get_hyperbolic_curvatures()
    return {}


# --- Lorentz residual heads ------------------------------------------------


def _supports_lorentz_heads(model):
    return _has_method(model, "add_lorentz_head")


def _set_lorentz_heads_trainable(model, layer_indices, trainable):
    if _has_method(model, "set_lorentz_heads_trainable"):
        model.set_lorentz_heads_trainable(layer_indices, trainable)
        return
    heads = getattr(model, "_lorentz_heads", None)
    if heads is None:
        return
    keys = (
        list(heads.keys())
        if layer_indices is None
        else [str(int(i)) for i in layer_indices]
    )
    for key in keys:
        if key in heads:
            for p in heads[key].parameters():
                p.requires_grad = bool(trainable)


def _get_lorentz_gates(model):
    if _has_method(model, "get_lorentz_gates"):
        return model.get_lorentz_gates()
    return dict(getattr(model, "_lorentz_gates", {}) or {})


def _set_lorentz_gate(model, layer_idx, gate):
    if _has_method(model, "set_lorentz_gate"):
        model.set_lorentz_gate(layer_idx, gate)
        return
    gates = getattr(model, "_lorentz_gates", None)
    if gates is None:
        raise RuntimeError(
            f"Cannot set Lorentz gate on layer {layer_idx}: encoder does not "
            "expose _lorentz_gates and has no set_lorentz_gate()"
        )
    gates[int(layer_idx)] = max(0.0, float(gate))


def _get_lorentz_curvatures(model):
    """Return {layer_idx_str: curvature_float} for all Lorentz heads."""
    if _has_method(model, "get_lorentz_curvatures"):
        return model.get_lorentz_curvatures()
    heads = getattr(model, "_lorentz_heads", None)
    if heads is None:
        return {}
    out = {}
    for key, head in heads.items():
        if hasattr(head, "curvature"):
            try:
                out[key] = float(head.curvature().detach().cpu())
            except Exception:
                continue
    return out


def _get_lorentz_head_curvature(model, layer_idx):
    """Scalar curvature of one Lorentz head, or None if unavailable."""
    heads = getattr(model, "_lorentz_heads", None)
    if heads is None or str(int(layer_idx)) not in heads:
        return None
    head = heads[str(int(layer_idx))]
    if hasattr(head, "curvature"):
        return float(head.curvature().detach().cpu())
    return None


def _set_lorentz_curvature(model, layer_idx, value):
    """Set curvature of one Lorentz head across all encoder variants.

    Order of preference:
      1. encoder-level ``set_lorentz_curvature`` (original encoder);
      2. direct ``head.set_curvature`` on the attached head (both the original
         and the Lorentz-only encoder have this);
      3. encoder-level ``set_curvature`` (Lorentz-only encoder's schedule, last
         resort — note this updates the schedule, not the head, so it is only
         used if the head is not reachable).
    """
    if _has_method(model, "set_lorentz_curvature"):
        model.set_lorentz_curvature(layer_idx, value)
        return
    heads = getattr(model, "_lorentz_heads", None)
    if heads is not None and str(int(layer_idx)) in heads:
        head = heads[str(int(layer_idx))]
        if hasattr(head, "set_curvature"):
            head.set_curvature(value)
            return
    if _has_method(model, "set_curvature"):
        model.set_curvature(layer_idx, value)
        return
    raise RuntimeError(
        f"Cannot set Lorentz curvature on layer {layer_idx}: encoder does not "
        "expose set_lorentz_curvature / set_curvature / head.set_curvature"
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def make_optimizer(model, adamw_lr, muon_lr, optimizer_cfg):
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()

    if optimizer_type == "muon":
        muon_params = []
        adamw_params = []
        exclude_embeddings = bool(optimizer_cfg.get("exclude_embeddings", True))
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_embedding = "embedding" in name.lower()
            if param.ndim == 2 and not (exclude_embeddings and is_embedding):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=float(optimizer_cfg.get("momentum", 0.95)),
            nesterov=bool(optimizer_cfg.get("nesterov", True)),
            ns_steps=int(optimizer_cfg.get("ns_steps", 5)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
        adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=tuple(optimizer_cfg.get("adamw_betas", (0.9, 0.999))),
            eps=float(optimizer_cfg.get("adamw_eps", 1.0e-8)),
            weight_decay=float(optimizer_cfg.get("adamw_weight_decay", 0.0)),
        )
        optimizer = MuonAdamW(muon=muon, adamw=adamw)
    elif optimizer_type == "adamw":
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=adamw_lr,
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
            eps=float(optimizer_cfg.get("eps", 1.0e-8)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return BasicOptimizer(
        model=model,
        optimizer=optimizer,
        clip_grad_threshold=optimizer_cfg.get("clip_grad_threshold", None),
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def identity_ratio_loss(outputs, mask, eps=1e-8):
    """Relative identity loss at the stacking boundary.

    It is zero iff the output of the newly stacked lower group (after its
    trainable boundary scalar) equals the representation that the pretrained
    subnet used to receive.  Minimising it drives the output/input ratio to 1.
    """
    aux_input = outputs.get("aux_input")
    aux_output = outputs.get("aux_output")
    if aux_input is None or aux_output is None:
        return None

    mask = mask.to(dtype=torch.bool)
    # The pretrained-subnet input is the reference: the auxiliary loss only
    # trains the newly stacked lower group and its boundary scalar.
    x = aux_input[mask].detach()
    y = aux_output[mask]
    if x.numel() == 0:
        return aux_input.new_zeros(())

    diff_sq = (y - x).pow(2).sum(dim=-1)
    reference_sq = x.pow(2).sum(dim=-1) + eps
    return (diff_sq / reference_sq).mean()


@torch.no_grad()
def identity_projection_ratio(outputs, mask, eps=1e-8):
    """Least-squares ratio of the boundary output to the input.

    `r = <y, x> / <x, x>` is the scalar that best maps the pretrained-subnet
    input to the new lower-stack output.  It is 1 when the ratio is exactly
    the identity.  (A plain norm ratio is not informative here because both
    representations are LayerNorm-ed and have the same norm.)
    """
    aux_input = outputs.get("aux_input")
    aux_output = outputs.get("aux_output")
    if aux_input is None or aux_output is None:
        return None

    mask = mask.to(dtype=torch.bool)
    x = aux_input[mask]
    y = aux_output[mask]
    if x.numel() == 0:
        return aux_input.new_ones(())

    numerator = (y * x).sum(dim=-1)
    denominator = (x * x).sum(dim=-1) + eps
    return (numerator / denominator).mean()


def hyperbolic_head_loss(outputs, labels):
    """Mean next-item CE over all active hyperbolic heads.

    The function is robust to encoders that never produce
    ``hyperbolic_logits_*`` (Euclidean and Lorentz-only variants): in that
    case it returns ``(None, {})`` and the training loop simply skips the
    auxiliary term.
    """
    losses = {}
    for key, logits in outputs.items():
        if key.startswith("hyperbolic_logits_"):
            layer_idx = int(key.rsplit("_", 1)[-1])
            losses[layer_idx] = F.cross_entropy(logits, labels)
    if not losses:
        return None, {}
    return sum(losses.values()) / len(losses), losses


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    train_dataloader,
    model,
    optimizer,
    loss_function,
    device,
    aux_coeff=0.0,
    aux_eps=1e-8,
    hyp_coeff=0.0,
):
    model.train()
    totals = {}
    steps = 0

    for batch in train_dataloader:
        move_batch(batch, device)
        outputs = model(batch)
        loss_inputs = {**batch, **outputs}
        rec_loss = loss_function(loss_inputs)
        total_loss = rec_loss
        batch_stats = {
            "loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
        }

        if "aux_input" in outputs and "aux_output" in outputs:
            if aux_coeff > 0.0:
                aux_loss = identity_ratio_loss(
                    outputs, batch["mask"], eps=aux_eps
                )
                total_loss = rec_loss + aux_coeff * aux_loss
            else:
                # Diagnostics only (auxiliary weight has decayed to zero).
                with torch.no_grad():
                    aux_loss = identity_ratio_loss(
                        outputs, batch["mask"], eps=aux_eps
                    )
            batch_stats = {
                "loss": total_loss.detach(),
                "rec_loss": rec_loss.detach(),
                "aux_loss": aux_loss.detach(),
                "identity_ratio": identity_projection_ratio(
                    outputs, batch["mask"], eps=aux_eps
                ),
            }

        hyp_loss, hyp_parts = hyperbolic_head_loss(
            outputs, batch["labels.ids"]
        )
        if hyp_loss is not None:
            if hyp_coeff > 0.0:
                total_loss = total_loss + hyp_coeff * hyp_loss
            batch_stats["hyperbolic_loss"] = hyp_loss.detach()
            for layer_idx, value in hyp_parts.items():
                batch_stats[f"hyperbolic_loss_{layer_idx}"] = value.detach()

        batch_stats["loss"] = total_loss.detach()
        optimizer.step(total_loss)

        for key, value in batch_stats.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1

    if steps == 0:
        return {}
    return {key: float(value / steps) for key, value in totals.items()}


def evaluate(val_dataloader, test_dataloader, model, metrics, device):
    val_metrics = inference(val_dataloader, model, metrics, device)
    test_metrics = inference(test_dataloader, model, metrics, device)
    return val_metrics, test_metrics


# ---------------------------------------------------------------------------
# Ramps and schedules
# ---------------------------------------------------------------------------


def stage_progress(stage_cfg, epoch):
    """Fraction of the new-group ramp completed at a stage-local epoch."""
    new_layers = list(stage_cfg.get("new_layers", []))
    ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))
    if not new_layers or ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(ramp_epochs))


def ramp_fraction(epoch, ramp_epochs):
    """Linear 0 -> 1 ramp that is exactly zero at stage epoch 0.

    Unlike :func:`stage_progress` (which uses ``(epoch + 1) / ramp`` and thus
    already starts at ``1 / ramp``), this is used for quantities that must be
    *exactly absent* at the first epoch of a stage: the Lorentz residual gate
    and the curvature.
    """
    ramp_epochs = int(ramp_epochs)
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(ramp_epochs))


def stage_scales(stage_cfg, epoch):
    """Return per-layer coefficients for a given stage-local epoch."""
    final_scales = [float(scale) for scale in stage_cfg["final_scales"]]
    new_layers = list(stage_cfg.get("new_layers", []))
    ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))

    if not new_layers or ramp_epochs <= 0:
        return final_scales, True

    fraction = stage_progress(stage_cfg, epoch)
    scales = list(final_scales)
    for layer_idx in new_layers:
        scales[layer_idx] = final_scales[layer_idx] * fraction
    complete = all(
        abs(scales[idx] - final_scales[idx]) < 1e-12
        for idx in range(len(scales))
    )
    return scales, complete


def aux_loss_coefficient(aux_cfg, alpha):
    """Coefficient of the boundary identity loss.

    Theoretical schedule: the growing network coefficient `alpha` and the
    falling auxiliary coefficient `lambda` satisfy the linear complementarity
    relation
        alpha + lambda / lambda0 = 1
    so
        lambda(alpha) = lambda0 * (1 - alpha).
    Since `alpha` is ramped linearly, `lambda` decays linearly to zero.
    `power` generalises this to (1 - alpha) ** power: power < 1 keeps the
    auxiliary weight higher for longer, power > 1 makes it decay faster.
    """
    if not aux_cfg or not bool(aux_cfg.get("enabled", False)):
        return 0.0
    lambda0 = float(aux_cfg.get("initial_coeff", 1.0))
    power = float(aux_cfg.get("power", 1.0))
    return lambda0 * max(0.0, 1.0 - float(alpha)) ** power


# ---------------------------------------------------------------------------
# One stage
# ---------------------------------------------------------------------------


def run_stage(
    stage_cfg,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_function,
    metrics,
    device,
    optimizer_cfg,
    global_epoch_start,
    aux_cfg=None,
    hyp_cfg=None,
    hyp_res_cfg=None,
):
    stage_name = stage_cfg["name"]
    num_layers = model.num_layers
    active_layers = [int(idx) for idx in stage_cfg["active_layers"]]
    new_layers = [int(idx) for idx in stage_cfg.get("new_layers", [])]
    final_scales = [float(scale) for scale in stage_cfg["final_scales"]]
    epochs = int(stage_cfg["epochs"])
    ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))

    aux_enabled = bool(aux_cfg and aux_cfg.get("enabled", False)) and bool(
        new_layers
    )
    aux_eps = float(aux_cfg.get("eps", 1e-8)) if aux_cfg else 1e-8
    aux_lambda0 = (
        float(aux_cfg.get("initial_coeff", 1.0)) if aux_cfg else 0.0
    )
    aux_power = float(aux_cfg.get("power", 1.0)) if aux_cfg else 1.0
    aux_lower_start = min(new_layers) if aux_enabled else None
    aux_upper_start = max(new_layers) + 1 if aux_enabled else None
    aux_scalar_layer = aux_upper_start - 1 if aux_enabled else None

    # A head is attached only when the stage actually stacks a new group; the
    # first stage is the plain 2-layer base model and must stay comparable to
    # the non-hyperbolic StackRec runs.
    hyp_enabled = (
        bool(hyp_cfg and hyp_cfg.get("enabled", False)) and bool(new_layers)
    )
    if hyp_enabled and not _supports_hyperbolic_heads(model):
        logger.warning(
            f"STAGE {stage_name}: config enables hyperbolic_heads, but the "
            "encoder does not expose add_hyperbolic_head(); disabling."
        )
        hyp_enabled = False

    hyp_head_layer = None
    hyp_depth = None
    hyp_curvature = None
    hyp_loss_coeff = 0.0
    hyp_warmup = True
    if hyp_enabled:
        # The head sits at the output of the group that has just been stacked.
        hyp_head_layer = max(new_layers)
        depth_mode = str(hyp_cfg.get("depth_mode", "absolute")).lower()
        if depth_mode == "absolute":
            hyp_depth = hyp_head_layer + 1
        elif depth_mode == "cumulative":
            hyp_depth = len(active_layers)
        elif depth_mode == "active_prefix":
            hyp_depth = sum(1 for idx in active_layers if idx <= hyp_head_layer)
        else:
            raise ValueError(f"Unknown hyperbolic depth_mode: {depth_mode}")

        schedule = str(hyp_cfg.get("schedule", "inverse_power")).lower()
        reference_curvature = float(hyp_cfg.get("reference_curvature", 1.0))
        reference_depth = float(hyp_cfg.get("reference_depth", 2.0))
        curvature_power = float(hyp_cfg.get("power", 2.0))
        if schedule == "constant":
            hyp_curvature = reference_curvature
        elif schedule == "inverse_power":
            hyp_curvature = reference_curvature * (
                reference_depth / max(float(hyp_depth), 1.0)
            ) ** curvature_power
        else:
            raise ValueError(f"Unknown hyperbolic curvature schedule: {schedule}")

        hyp_loss_coeff = float(hyp_cfg.get("loss_coeff", 0.1))
        hyp_warmup = bool(hyp_cfg.get("warmup", True))

    # Lorentz hyperbolic-residual heads.  They are attached to the output of
    # every newly copied (stacked) group.  Both the residual gate and the
    # curvature are ramped from exactly zero, so at the moment a group is
    # stacked the model is still exactly the pretrained subnet.
    hyp_res_enabled = (
        bool(hyp_res_cfg and hyp_res_cfg.get("enabled", False))
        and bool(new_layers)
    )
    if hyp_res_enabled and not _supports_lorentz_heads(model):
        logger.warning(
            f"STAGE {stage_name}: config enables hyperbolic_residual, but the "
            "encoder does not expose add_lorentz_head(); disabling."
        )
        hyp_res_enabled = False

    hyp_res_head_layer = None
    hyp_res_curvature_target = None
    hyp_res_residual_ramp = 0
    hyp_res_curvature_ramp = 0
    hyp_res_loss_coeff = 0.0
    hyp_res_warmup = True
    if hyp_res_enabled:
        hyp_res_head_layer = max(new_layers)
        curvature_targets = hyp_res_cfg.get("curvature_targets", {}) or {}
        hyp_res_curvature_target = float(
            curvature_targets.get(
                stage_name,
                stage_cfg.get(
                    "hyperbolic_curvature_target",
                    hyp_res_cfg.get("default_curvature_target", 1.0),
                ),
            )
        )
        default_ramp = max(int(ramp_epochs), 1)
        hyp_res_residual_ramp = int(
            hyp_res_cfg.get("residual_ramp_epochs", default_ramp)
        )
        hyp_res_curvature_ramp = int(
            hyp_res_cfg.get("curvature_ramp_epochs", default_ramp)
        )
        hyp_res_loss_coeff = float(hyp_res_cfg.get("loss_coeff", 0.1))
        hyp_res_warmup = bool(hyp_res_cfg.get("warmup", True))

    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type == "muon":
        lr = None
        muon_lr = float(stage_cfg["muon_lr"])
        adamw_lr = float(stage_cfg["adamw_lr"])
        lr_description = f"muon={muon_lr}, adamw={adamw_lr}"
    else:
        lr = float(stage_cfg["lr"])
        muon_lr = None
        adamw_lr = lr
        lr_description = f"adamw={adamw_lr}"

    logger.info("=" * 80)
    logger.info(
        f"STAGE {stage_name}: active_layers={active_layers}, {lr_description}"
    )
    logger.info(
        f"STAGE {stage_name}: epochs={epochs}, ramp_epochs={ramp_epochs}, "
        f"new_layers={new_layers}, final_scales={final_scales}"
    )

    for src_idx, dst_idx in stage_cfg.get("copy_layers", []):
        src_idx, dst_idx = int(src_idx), int(dst_idx)
        model.copy_layer_weights(src_idx, dst_idx)
        logger.info(
            f"STAGE {stage_name}: copied layer {src_idx} -> layer {dst_idx}"
        )

    model.set_layers_trainable(range(num_layers), False)
    model.set_layers_trainable(active_layers, True)

    if aux_enabled:
        # The scalar sits on the last layer of the newly stacked lower group,
        # i.e. exactly at the interface with the pretrained upper subnet.
        model.enable_layer_aux_scalar(
            aux_scalar_layer, init=float(aux_cfg.get("scalar_init", 1.0))
        )
        model.set_aux_boundary(aux_lower_start, aux_upper_start)
        logger.info(
            f"STAGE {stage_name}: aux identity loss on layers "
            f"[{aux_lower_start}, {aux_upper_start}) "
            f"(scalar on layer {aux_scalar_layer}), "
            f"lambda(alpha) = {aux_lambda0:.6g} * (1 - alpha)^{aux_power:g}"
        )
    else:
        model.clear_aux_boundary()

    if hyp_enabled:
        if bool(hyp_cfg.get("freeze_previous", True)):
            _set_hyperbolic_heads_trainable(model, None, False)
        head = model.add_hyperbolic_head(
            hyp_head_layer,
            **_filter_kwargs(
                model.add_hyperbolic_head,
                curvature=hyp_curvature,
                learnable_curvature=bool(
                    hyp_cfg.get("learnable_curvature", False)
                ),
                temperature=float(hyp_cfg.get("temperature", 0.1)),
                distance=str(hyp_cfg.get("distance", "squared")),
            ),
        )
        _set_hyperbolic_heads_trainable(model, [hyp_head_layer], True)
        logger.info(
            f"STAGE {stage_name}: hyperbolic head on layer {hyp_head_layer} "
            f"(depth={hyp_depth}, schedule="
            f"{hyp_cfg.get('schedule', 'inverse_power')}, "
            f"c={float(head.curvature().detach().cpu()):.6f}, "
            f"learnable={bool(hyp_cfg.get('learnable_curvature', False))}), "
            f"loss_coeff={hyp_loss_coeff:g}"
        )

    if hyp_res_enabled:
        if bool(hyp_res_cfg.get("freeze_previous", True)):
            _set_lorentz_heads_trainable(model, None, False)
        model.add_lorentz_head(
            hyp_res_head_layer,
            **_filter_kwargs(
                model.add_lorentz_head,
                curvature=0.0,
                learnable_curvature=bool(
                    hyp_res_cfg.get("learnable_curvature", False)
                ),
                temperature=float(hyp_res_cfg.get("temperature", 0.1)),
                distance=str(hyp_res_cfg.get("distance", "squared")),
            ),
        )
        _set_lorentz_heads_trainable(model, [hyp_res_head_layer], True)
        _set_lorentz_gate(model, hyp_res_head_layer, 0.0)
        _set_lorentz_curvature(model, hyp_res_head_layer, 0.0)
        logger.info(
            f"STAGE {stage_name}: Lorentz residual head on layer "
            f"{hyp_res_head_layer} (curvature 0 -> "
            f"{hyp_res_curvature_target:g} over "
            f"{hyp_res_curvature_ramp} epochs, residual gate 0 -> 1 over "
            f"{hyp_res_residual_ramp} epochs, "
            f"loss_coeff={hyp_res_loss_coeff:g})"
        )

    # Sanity check: before the ramp starts, newly activated layers contribute
    # exactly zero, so the model must reproduce the previous stage.
    stage_start_val_metrics = None
    stage_start_test_metrics = None
    if new_layers:
        start_scales = list(final_scales)
        for layer_idx in new_layers:
            start_scales[layer_idx] = 0.0
        model.set_layer_scales(start_scales)
        if hyp_res_enabled:
            _set_lorentz_gate(model, hyp_res_head_layer, 0.0)
            _set_lorentz_curvature(model, hyp_res_head_layer, 0.0)
        stage_start_val_metrics, stage_start_test_metrics = evaluate(
            val_dataloader, test_dataloader, model, metrics, device
        )
        logger.info(
            f"STAGE {stage_name}: start (new gates=0) "
            f"val/ndcg@10={stage_start_val_metrics['ndcg@10']:.6f}, "
            f"test/ndcg@10={stage_start_test_metrics['ndcg@10']:.6f}"
        )

    optimizer = make_optimizer(
        model, adamw_lr=adamw_lr, muon_lr=muon_lr, optimizer_cfg=optimizer_cfg
    )
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    logger.info(f"STAGE {stage_name}: trainable params={trainable_params}")

    history = []
    best_complete_val = -float("inf")
    best_complete_epoch = None
    best_complete_global_epoch = None
    best_complete_scales = None
    best_complete_val_metrics = None
    best_complete_test_metrics = None
    best_complete_state = None
    best_complete_lorentz_gates = None

    best_any_val = -float("inf")
    best_any_epoch = None
    best_any_global_epoch = None
    best_any_scales = None
    best_any_val_metrics = None
    best_any_test_metrics = None

    final_scales_used = None
    final_val_metrics = None
    final_test_metrics = None

    stage_start_time = time.time()
    for epoch in range(epochs):
        scales, complete = stage_scales(stage_cfg, epoch)
        model.set_layer_scales(scales)
        final_scales_used = scales

        alpha = stage_progress(stage_cfg, epoch)
        aux_coeff = (
            aux_loss_coefficient(aux_cfg, alpha) if aux_enabled else 0.0
        )

        residual_gate = 0.0
        curvature_fraction = 0.0
        if hyp_res_enabled:
            residual_gate = ramp_fraction(epoch, hyp_res_residual_ramp)
            curvature_fraction = ramp_fraction(epoch, hyp_res_curvature_ramp)
            _set_lorentz_gate(model, hyp_res_head_layer, residual_gate)
            _set_lorentz_curvature(
                model,
                hyp_res_head_layer,
                hyp_res_curvature_target * curvature_fraction,
            )

        hyp_coeff_epoch = 0.0
        if hyp_enabled:
            hyp_coeff_epoch = max(
                hyp_coeff_epoch,
                hyp_loss_coeff * (alpha if hyp_warmup else 1.0),
            )
        if hyp_res_enabled:
            hyp_coeff_epoch = max(
                hyp_coeff_epoch,
                hyp_res_loss_coeff
                * (residual_gate if hyp_res_warmup else 1.0),
            )

        residual_complete = (
            not hyp_res_enabled
            or (
                residual_gate >= 1.0 - 1.0e-12
                and curvature_fraction >= 1.0 - 1.0e-12
            )
        )
        complete = complete and residual_complete

        train_stats = train_one_epoch(
            train_dataloader,
            model,
            optimizer,
            loss_function,
            device,
            aux_coeff=aux_coeff,
            aux_eps=aux_eps,
            hyp_coeff=hyp_coeff_epoch,
        )
        val_metrics, test_metrics = evaluate(
            val_dataloader, test_dataloader, model, metrics, device
        )

        global_epoch = global_epoch_start + epoch
        val_ndcg = float(val_metrics["ndcg@10"])
        test_ndcg = float(test_metrics["ndcg@10"])
        layer_aux_scalars = model.get_layer_aux_scalars()

        history.append(
            {
                "stage": stage_name,
                "stage_epoch": epoch,
                "global_epoch": global_epoch,
                "optimizer_type": optimizer_type,
                "lr": lr,
                "muon_lr": muon_lr,
                "adamw_lr": adamw_lr,
                "layer_scales": scales,
                "gates_complete": complete,
                "aux_enabled": aux_enabled,
                "aux_alpha": alpha,
                "aux_loss_coeff": aux_coeff,
                "layer_aux_scalars": layer_aux_scalars,
                "hyperbolic_head": {
                    "enabled": hyp_enabled,
                    "head_layer": hyp_head_layer,
                    "depth": hyp_depth,
                    "scheduled_curvature": hyp_curvature,
                    "loss_coeff": hyp_coeff_epoch,
                    "curvatures": _get_hyperbolic_curvatures(model),
                },
                "hyperbolic_residual": {
                    "enabled": hyp_res_enabled,
                    "head_layer": hyp_res_head_layer,
                    "curvature_target": hyp_res_curvature_target,
                    "curvature": _get_lorentz_head_curvature(
                        model, hyp_res_head_layer
                    )
                    if hyp_res_enabled
                    else None,
                    "curvature_fraction": curvature_fraction,
                    "residual_gate": residual_gate,
                    "loss_coeff": hyp_coeff_epoch if hyp_res_enabled else 0.0,
                    "gates": _get_lorentz_gates(model),
                    "curvatures": _get_lorentz_curvatures(model),
                },
                "train": train_stats,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

        aux_log = ""
        if aux_enabled:
            aux_log = (
                f" alpha={alpha:.3f} lambda={aux_coeff:.4f}"
                f" aux_loss={train_stats.get('aux_loss', float('nan')):.4f}"
                f" ratio={train_stats.get('identity_ratio', float('nan')):.4f}"
            )
        hyp_log = ""
        if hyp_enabled:
            hyp_head_c = float(
                model._hyperbolic_heads[str(hyp_head_layer)]
                .curvature()
                .detach()
                .cpu()
            )
            hyp_log = (
                f" hyp_layer={hyp_head_layer}"
                f" c={hyp_head_c:.4f}"
                f" hyp_coeff={hyp_coeff_epoch:.4f}"
                f" hyp_loss={train_stats.get('hyperbolic_loss', float('nan')):.4f}"
            )
        hyp_res_log = ""
        if hyp_res_enabled:
            head_c = _get_lorentz_head_curvature(model, hyp_res_head_layer)
            head_c_str = f"{head_c:.4f}" if head_c is not None else "nan"
            hyp_res_log = (
                f" res_layer={hyp_res_head_layer}"
                f" res_gate={residual_gate:.3f}"
                f" res_c={head_c_str}"
                f" res_coeff={hyp_coeff_epoch:.4f}"
                f" res_loss={train_stats.get('hyperbolic_loss', float('nan')):.4f}"
            )
        logger.info(
            f"STAGE {stage_name} epoch {epoch + 1}/{epochs} "
            f"scales={[round(scale, 3) for scale in scales]}"
            f"{aux_log}{hyp_log}{hyp_res_log} "
            f"val/ndcg@10={val_ndcg:.6f} test/ndcg@10={test_ndcg:.6f}"
        )

        if val_ndcg > best_any_val:
            best_any_val = val_ndcg
            best_any_epoch = epoch
            best_any_global_epoch = global_epoch
            best_any_scales = list(scales)
            best_any_val_metrics = val_metrics
            best_any_test_metrics = test_metrics

        if complete and val_ndcg > best_complete_val:
            best_complete_val = val_ndcg
            best_complete_epoch = epoch
            best_complete_global_epoch = global_epoch
            best_complete_scales = list(scales)
            best_complete_val_metrics = val_metrics
            best_complete_test_metrics = test_metrics
            best_complete_state = copy.deepcopy(model.state_dict())
            if hyp_res_enabled:
                best_complete_lorentz_gates = _get_lorentz_gates(model)

        final_val_metrics = val_metrics
        final_test_metrics = test_metrics

    stage_time = time.time() - stage_start_time

    if best_complete_state is not None:
        model.load_state_dict(best_complete_state)
        model.set_layer_scales(best_complete_scales)
        # Lorentz curvature buffers are part of the state dict, but the gates
        # are plain python floats, so restore them explicitly.
        if best_complete_lorentz_gates:
            lorentz_heads = getattr(model, "_lorentz_heads", {}) or {}
            for layer_idx, gate in best_complete_lorentz_gates.items():
                if str(int(layer_idx)) in lorentz_heads:
                    _set_lorentz_gate(model, int(layer_idx), gate)
        model.set_layers_trainable(range(num_layers), False)
        model.set_layers_trainable(active_layers, True)
        logger.info(
            f"STAGE {stage_name}: restored best complete checkpoint "
            f"(stage_epoch={best_complete_epoch}, "
            f"val/ndcg@10={best_complete_val:.6f})"
        )
    else:
        logger.warning(
            f"STAGE {stage_name}: no complete-gate epoch found, "
            "keeping the final weights"
        )

    summary = {
        "name": stage_name,
        "num_active_layers": len(active_layers),
        "active_layers": active_layers,
        "new_layers": new_layers,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "muon_lr": muon_lr,
        "adamw_lr": adamw_lr,
        "epochs": epochs,
        "ramp_epochs": ramp_epochs,
        "final_scales": final_scales,
        "copy_layers": stage_cfg.get("copy_layers", []),
        "trainable_params": trainable_params,
        "train_time_sec": stage_time,
        "global_epoch_start": global_epoch_start,
        "global_epoch_end": global_epoch_start + epochs - 1,
        "stage_start_val_metrics": stage_start_val_metrics,
        "stage_start_test_metrics": stage_start_test_metrics,
        "best_complete_epoch": best_complete_epoch,
        "best_complete_global_epoch": best_complete_global_epoch,
        "best_complete_layer_scales": best_complete_scales,
        "best_complete_val_metrics": best_complete_val_metrics,
        "best_complete_test_metrics": best_complete_test_metrics,
        "best_complete_key_metrics": {
            "val": select_key_metrics(best_complete_val_metrics),
            "test": select_key_metrics(best_complete_test_metrics),
        }
        if best_complete_val_metrics is not None
        else None,
        "best_any_epoch": best_any_epoch,
        "best_any_global_epoch": best_any_global_epoch,
        "best_any_layer_scales": best_any_scales,
        "best_any_val_metrics": best_any_val_metrics,
        "best_any_test_metrics": best_any_test_metrics,
        "final_layer_scales": final_scales_used,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "final_key_metrics": {
            "val": select_key_metrics(final_val_metrics),
            "test": select_key_metrics(final_test_metrics),
        },
        "aux_identity": {
            "enabled": aux_enabled,
            "lower_start": aux_lower_start,
            "upper_start": aux_upper_start,
            "scalar_layer": aux_scalar_layer,
            "initial_coeff": aux_lambda0,
            "power": aux_power,
            "eps": aux_eps,
            "relation": "lambda(alpha) = lambda0 * (1 - alpha) ** power",
            "layer_aux_scalars": model.get_layer_aux_scalars(),
        },
        "hyperbolic_head": {
            "enabled": hyp_enabled,
            "head_layer": hyp_head_layer,
            "depth": hyp_depth,
            "depth_mode": (
                str(hyp_cfg.get("depth_mode", "absolute")) if hyp_cfg else None
            ),
            "schedule": hyp_cfg.get("schedule") if hyp_cfg else None,
            "scheduled_curvature": hyp_curvature,
            "loss_coeff": hyp_loss_coeff,
            "warmup": hyp_warmup,
            "learnable_curvature": (
                bool(hyp_cfg.get("learnable_curvature", False))
                if hyp_cfg
                else False
            ),
            "curvatures": _get_hyperbolic_curvatures(model),
        },
        "hyperbolic_residual": {
            "enabled": hyp_res_enabled,
            "head_layer": hyp_res_head_layer,
            "curvature_target": hyp_res_curvature_target,
            "residual_ramp_epochs": hyp_res_residual_ramp,
            "curvature_ramp_epochs": hyp_res_curvature_ramp,
            "loss_coeff": hyp_res_loss_coeff,
            "warmup": hyp_res_warmup,
            "normalization": (
                "the residual is rescaled to the norm of the boundary state, "
                "added with a gate ramped from 0 to 1, and the sum is "
                "renormalised back to that norm"
            ),
            "gates": _get_lorentz_gates(model),
            "curvatures": _get_lorentz_curvatures(model),
        },
    }

    return summary, history


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="beauty_sasrec_stackrec",
)
def main(cfg):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["model"]["topk_k"] = max(config["metrics_ks"])
    OmegaConf.save(config, output_dir / "config.yaml")

    logger.info(f"StackRec config:\n{OmegaConf.to_yaml(config)}\n")

    fix_random_seed(config.get("seed", 42))
    device = config["device"]
    logger.info(f"Current DEVICE: {device}")

    dataloaders, dataset_meta = prepare_data(config, stage="select_epoch")

    item2num_interactions = None
    data_dir = Path(config["dataset"]["path_to_data_dir"])
    dataset_name = config["dataset"]["name"]
    train_path = data_dir / "global_split" / dataset_name / "train.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        item2num_interactions = train_df.groupby("item_id").size().to_dict()

    model = build_model(config["model_name"], config, dataset_meta).to(device)
    loss_function = instantiate(config["loss"])
    metrics = build_metrics(config, dataset_meta, item2num_interactions)

    logger.info(f"Num params: {sum(p.numel() for p in model.parameters())}")

    stage_summaries = []
    histories = {}
    global_epoch = 0
    aux_cfg = config["stackrec"].get("aux_identity", {})
    hyp_cfg = config["stackrec"].get("hyperbolic_heads", {})
    hyp_res_cfg = config["stackrec"].get("hyperbolic_residual", {})

    for stage_cfg in config["stackrec"]["stages"]:
        print(config)
        summary, history = run_stage(
            stage_cfg=stage_cfg,
            model=model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["validation"],
            test_dataloader=dataloaders["test"],
            loss_function=loss_function,
            metrics=metrics,
            device=device,
            optimizer_cfg=config["optimizer"],
            global_epoch_start=global_epoch,
            aux_cfg=aux_cfg,
            hyp_cfg=hyp_cfg,
            hyp_res_cfg=hyp_res_cfg,
        )
        stage_summaries.append(summary)
        histories[summary["name"]] = history
        global_epoch += int(stage_cfg["epochs"])

        with open(output_dir / f"{summary['name']}_best_metrics.json", "w") as f:
            json.dump(
                {
                    "best_complete": {
                        "val": summary["best_complete_val_metrics"],
                        "test": summary["best_complete_test_metrics"],
                    },
                    "final": {
                        "val": summary["final_val_metrics"],
                        "test": summary["final_test_metrics"],
                    },
                },
                f,
                indent=2,
            )

    results = {
        "description": (
            f"{config['optimizer'].get('type', 'adamw')} + StackRec on "
            "SASRec/Beauty. One continuous training run that progressively "
            "activates the last 2, middle 4 and all 6 transformer layers."
        ),
        "optimizer_type": config["optimizer"].get("type", "adamw"),
        "interpretation": {
            "layer_scales": (
                "Per-layer forward coefficient. A zero coefficient makes the "
                "layer an identity (it is skipped); newly activated layers are "
                "ramped from 0 to 1."
            ),
            "stages": {
                "2_layers": "layers 4,5 trained; layers 0-3 frozen with scale 0",
                "4_layers": "layer 4->2 and 5->3; layers 2-5 trained; scales of 2,3 ramped",
                "6_layers": "layer 2->0 and 3->1; all 6 layers trained; scales of 0,1 ramped",
            },
            "aux_identity": (
                "When a new group is stacked below the trained subnet, the "
                "output scalar of its last layer becomes trainable and an "
                "auxiliary loss drives ||g * output|| / ||input|| to 1 at the "
                "interface. Its weight follows the linear-complement law "
                "lambda(alpha) = lambda0 * (1 - alpha), so it decays linearly "
                "while the new-group scale alpha ramps from 0 to 1."
            ),
            "selection": (
                "best_complete_* are metrics at the best validation ndcg@10 "
                "among epochs where all ramp coefficients reached their final "
                "values. final_* are metrics at the last epoch of the stage."
            ),
            "hyperbolic_heads": (
                "Optional Poincare-ball next-item heads attached to the output "
                "of each newly stacked group. The boundary representation and "
                "the item table are mapped through expmap0 and scored by "
                "negative hyperbolic distance. Curvature follows "
                "c(depth) = c_ref * (depth_ref / depth) ** power (or is "
                "learned), so deeper/upper heads use a flatter, larger ball. "
                "Only available when the encoder exposes add_hyperbolic_head(); "
                "otherwise the branch is disabled with a warning."
            ),
            "hyperbolic_residual": (
                "Lorentz/hyperboloid residual heads attached to the output of "
                "every copied group. The residual is the curvature-induced "
                "correction of the Lorentz exponential map, so it is exactly "
                "zero at c=0. It is multiplied by a gate ramped from 0 to 1, "
                "rescaled to the norm of the boundary state, added, and the "
                "sum is renormalised back to that norm. Thus at the moment a "
                "group is stacked the model is exactly the pretrained subnet, "
                "and the input norm of the pretrained subnet is preserved "
                "throughout the ramp. Curvature ramps from 0 to the stage "
                "target (0.2 for the first copy, 1.0 for the second copy)."
            ),
        },
        "config": config,
        "stages": {summary["name"]: summary for summary in stage_summaries},
        "summary": {
            summary["name"]: {
                "num_active_layers": summary["num_active_layers"],
                "optimizer_type": summary["optimizer_type"],
                "lr": summary["lr"],
                "muon_lr": summary["muon_lr"],
                "adamw_lr": summary["adamw_lr"],
                "epochs": summary["epochs"],
                "best_complete_epoch": summary["best_complete_epoch"],
                "best_complete_global_epoch": summary[
                    "best_complete_global_epoch"
                ],
                "best_complete": summary["best_complete_key_metrics"],
                "final": summary["final_key_metrics"],
                "aux_identity": summary["aux_identity"],
                "hyperbolic_head": summary["hyperbolic_head"],
                "hyperbolic_residual": summary["hyperbolic_residual"],
            }
            for summary in stage_summaries
        },
    }

    with open(output_dir / "stackrec_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / config["stackrec"]["history_path"], "w") as f:
        json.dump(histories, f, indent=2)

    results_path = Path(config["stackrec"]["results_path"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved stage results to {output_dir / 'stackrec_results.json'}")
    logger.info(f"Saved history to {output_dir / config['stackrec']['history_path']}")
    logger.info(f"Saved summary to {results_path}")

    for summary in stage_summaries:
        logger.info(
            f"RESULT {summary['name']}: "
            f"val/ndcg@10={summary['best_complete_val_metrics']['ndcg@10']:.6f}, "
            f"test/ndcg@10={summary['best_complete_test_metrics']['ndcg@10']:.6f}, "
            f"val/recall@10={summary['best_complete_val_metrics']['recall@10']:.6f}, "
            f"test/recall@10={summary['best_complete_test_metrics']['recall@10']:.6f}"
        )


if __name__ == "__main__":
    main()

# """AdamW + StackRec for SASRec.

# The experiment starts from a 6-layer SASRec and progressively activates its
# layers:

#     last 2 layers  ->  middle 4 layers  ->  all 6 layers

# New layers are initialised from already trained layers and are introduced
# through a per-layer forward coefficient that is ramped from 0 to 1.  The whole
# procedure is run as one training process; validation/test metrics are recorded
# after every stage.

# When a new group is stacked, an extra trainable scalar is attached to the last
# layer of that group and an auxiliary identity loss keeps the representation
# entering the old subnet close to the representation it received before the
# stacking.  The auxiliary weight follows
#     lambda(alpha) = lambda0 * (1 - alpha),
# where alpha is the growing forward coefficient of the new group.
# """

# import copy
# import json
# import time
# from pathlib import Path

# import hydra
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import instantiate
# from omegaconf import OmegaConf

# from run_model import build_metrics, build_model, prepare_data
# from src.optimizer import BasicOptimizer, Muon, MuonAdamW
# from src.utils import create_logger, fix_random_seed, inference, move_batch

# logger = create_logger(name=__name__)


# KEY_METRICS = [
#     "ndcg@10",
#     "recall@10",
#     "ndcg@100",
#     "recall@100",
#     "coverage@10",
#     "coverage@100",
#     "meanrank@10",
#     "meanrank@100",
# ]


# def select_key_metrics(metrics):
#     return {name: metrics[name] for name in KEY_METRICS if name in metrics}


# def make_optimizer(model, adamw_lr, muon_lr, optimizer_cfg):
#     optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()

#     if optimizer_type == "muon":
#         muon_params = []
#         adamw_params = []
#         exclude_embeddings = bool(optimizer_cfg.get("exclude_embeddings", True))
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue
#             is_embedding = "embedding" in name.lower()
#             if param.ndim == 2 and not (exclude_embeddings and is_embedding):
#                 muon_params.append(param)
#             else:
#                 adamw_params.append(param)

#         muon = Muon(
#             muon_params,
#             lr=muon_lr,
#             momentum=float(optimizer_cfg.get("momentum", 0.95)),
#             nesterov=bool(optimizer_cfg.get("nesterov", True)),
#             ns_steps=int(optimizer_cfg.get("ns_steps", 5)),
#             weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
#         )
#         adamw = torch.optim.AdamW(
#             adamw_params,
#             lr=adamw_lr,
#             betas=tuple(optimizer_cfg.get("adamw_betas", (0.9, 0.999))),
#             eps=float(optimizer_cfg.get("adamw_eps", 1.0e-8)),
#             weight_decay=float(optimizer_cfg.get("adamw_weight_decay", 0.0)),
#         )
#         optimizer = MuonAdamW(muon=muon, adamw=adamw)
#     elif optimizer_type == "adamw":
#         params = [param for param in model.parameters() if param.requires_grad]
#         optimizer = torch.optim.AdamW(
#             params,
#             lr=adamw_lr,
#             betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
#             eps=float(optimizer_cfg.get("eps", 1.0e-8)),
#             weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
#         )
#     else:
#         raise ValueError(f"Unknown optimizer type: {optimizer_type}")

#     return BasicOptimizer(
#         model=model,
#         optimizer=optimizer,
#         clip_grad_threshold=optimizer_cfg.get("clip_grad_threshold", None),
#     )


# def identity_ratio_loss(outputs, mask, eps=1e-8):
#     """Relative identity loss at the stacking boundary.

#     It is zero iff the output of the newly stacked lower group (after its
#     trainable boundary scalar) equals the representation that the pretrained
#     subnet used to receive.  Minimising it drives the output/input ratio to 1.
#     """
#     aux_input = outputs.get("aux_input")
#     aux_output = outputs.get("aux_output")
#     if aux_input is None or aux_output is None:
#         return None

#     mask = mask.to(dtype=torch.bool)
#     # The pretrained-subnet input is the reference: the auxiliary loss only
#     # trains the newly stacked lower group and its boundary scalar.
#     x = aux_input[mask].detach()
#     y = aux_output[mask]
#     if x.numel() == 0:
#         return aux_input.new_zeros(())

#     diff_sq = (y - x).pow(2).sum(dim=-1)
#     reference_sq = x.pow(2).sum(dim=-1) + eps
#     return (diff_sq / reference_sq).mean()


# @torch.no_grad()
# def identity_projection_ratio(outputs, mask, eps=1e-8):
#     """Least-squares ratio of the boundary output to the input.

#     `r = <y, x> / <x, x>` is the scalar that best maps the pretrained-subnet
#     input to the new lower-stack output.  It is 1 when the ratio is exactly
#     the identity.  (A plain norm ratio is not informative here because both
#     representations are LayerNorm-ed and have the same norm.)
#     """
#     aux_input = outputs.get("aux_input")
#     aux_output = outputs.get("aux_output")
#     if aux_input is None or aux_output is None:
#         return None

#     mask = mask.to(dtype=torch.bool)
#     x = aux_input[mask]
#     y = aux_output[mask]
#     if x.numel() == 0:
#         return aux_input.new_ones(())

#     numerator = (y * x).sum(dim=-1)
#     denominator = (x * x).sum(dim=-1) + eps
#     return (numerator / denominator).mean()


# def hyperbolic_head_loss(outputs, labels):
#     """Mean next-item CE over all active hyperbolic heads."""
#     losses = {}
#     for key, logits in outputs.items():
#         if key.startswith("hyperbolic_logits_"):
#             layer_idx = int(key.rsplit("_", 1)[-1])
#             losses[layer_idx] = F.cross_entropy(logits, labels)
#     if not losses:
#         return None, {}
#     return sum(losses.values()) / len(losses), losses


# def train_one_epoch(
#     train_dataloader,
#     model,
#     optimizer,
#     loss_function,
#     device,
#     aux_coeff=0.0,
#     aux_eps=1e-8,
#     hyp_coeff=0.0,
# ):
#     model.train()
#     totals = {}
#     steps = 0

#     for batch in train_dataloader:
#         move_batch(batch, device)
#         outputs = model(batch)
#         loss_inputs = {**batch, **outputs}
#         rec_loss = loss_function(loss_inputs)
#         total_loss = rec_loss
#         batch_stats = {
#             "loss": total_loss.detach(),
#             "rec_loss": rec_loss.detach(),
#         }

#         if "aux_input" in outputs and "aux_output" in outputs:
#             if aux_coeff > 0.0:
#                 aux_loss = identity_ratio_loss(
#                     outputs, batch["mask"], eps=aux_eps
#                 )
#                 total_loss = rec_loss + aux_coeff * aux_loss
#             else:
#                 # Diagnostics only (auxiliary weight has decayed to zero).
#                 with torch.no_grad():
#                     aux_loss = identity_ratio_loss(
#                         outputs, batch["mask"], eps=aux_eps
#                     )
#             batch_stats = {
#                 "loss": total_loss.detach(),
#                 "rec_loss": rec_loss.detach(),
#                 "aux_loss": aux_loss.detach(),
#                 "identity_ratio": identity_projection_ratio(
#                     outputs, batch["mask"], eps=aux_eps
#                 ),
#             }

#         hyp_loss, hyp_parts = hyperbolic_head_loss(
#             outputs, batch["labels.ids"]
#         )
#         if hyp_loss is not None:
#             if hyp_coeff > 0.0:
#                 total_loss = total_loss + hyp_coeff * hyp_loss
#             batch_stats["hyperbolic_loss"] = hyp_loss.detach()
#             for layer_idx, value in hyp_parts.items():
#                 batch_stats[f"hyperbolic_loss_{layer_idx}"] = value.detach()

#         batch_stats["loss"] = total_loss.detach()
#         optimizer.step(total_loss)

#         for key, value in batch_stats.items():
#             totals[key] = totals.get(key, 0.0) + value
#         steps += 1

#     if steps == 0:
#         return {}
#     return {key: float(value / steps) for key, value in totals.items()}


# def evaluate(val_dataloader, test_dataloader, model, metrics, device):
#     val_metrics = inference(val_dataloader, model, metrics, device)
#     test_metrics = inference(test_dataloader, model, metrics, device)
#     return val_metrics, test_metrics


# def stage_progress(stage_cfg, epoch):
#     """Fraction of the new-group ramp completed at a stage-local epoch."""
#     new_layers = list(stage_cfg.get("new_layers", []))
#     ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))
#     if not new_layers or ramp_epochs <= 0:
#         return 1.0
#     return min(1.0, float(epoch + 1) / float(ramp_epochs))


# def ramp_fraction(epoch, ramp_epochs):
#     """Linear 0 -> 1 ramp that is exactly zero at stage epoch 0.

#     Unlike :func:`stage_progress` (which uses ``(epoch + 1) / ramp`` and thus
#     already starts at ``1 / ramp``), this is used for quantities that must be
#     *exactly absent* at the first epoch of a stage: the Lorentz residual gate
#     and the curvature.
#     """
#     ramp_epochs = int(ramp_epochs)
#     if ramp_epochs <= 0:
#         return 1.0
#     return min(1.0, float(epoch) / float(ramp_epochs))


# def stage_scales(stage_cfg, epoch):
#     """Return per-layer coefficients for a given stage-local epoch."""
#     final_scales = [float(scale) for scale in stage_cfg["final_scales"]]
#     new_layers = list(stage_cfg.get("new_layers", []))
#     ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))

#     if not new_layers or ramp_epochs <= 0:
#         return final_scales, True

#     fraction = stage_progress(stage_cfg, epoch)
#     scales = list(final_scales)
#     for layer_idx in new_layers:
#         scales[layer_idx] = final_scales[layer_idx] * fraction
#     complete = all(abs(scales[idx] - final_scales[idx]) < 1e-12 for idx in range(len(scales)))
#     return scales, complete


# def aux_loss_coefficient(aux_cfg, alpha):
#     """Coefficient of the boundary identity loss.

#     Theoretical schedule: the growing network coefficient `alpha` and the
#     falling auxiliary coefficient `lambda` satisfy the linear complementarity
#     relation
#         alpha + lambda / lambda0 = 1
#     so
#         lambda(alpha) = lambda0 * (1 - alpha).
#     Since `alpha` is ramped linearly, `lambda` decays linearly to zero.
#     `power` generalises this to (1 - alpha) ** power: power < 1 keeps the
#     auxiliary weight higher for longer, power > 1 makes it decay faster.
#     """
#     if not aux_cfg or not bool(aux_cfg.get("enabled", False)):
#         return 0.0
#     lambda0 = float(aux_cfg.get("initial_coeff", 1.0))
#     power = float(aux_cfg.get("power", 1.0))
#     return lambda0 * max(0.0, 1.0 - float(alpha)) ** power


# def run_stage(
#     stage_cfg,
#     model,
#     train_dataloader,
#     val_dataloader,
#     test_dataloader,
#     loss_function,
#     metrics,
#     device,
#     optimizer_cfg,
#     global_epoch_start,
#     aux_cfg=None,
#     hyp_cfg=None,
#     hyp_res_cfg=None,
# ):
#     stage_name = stage_cfg["name"]
#     num_layers = model.num_layers
#     active_layers = [int(idx) for idx in stage_cfg["active_layers"]]
#     new_layers = [int(idx) for idx in stage_cfg.get("new_layers", [])]
#     final_scales = [float(scale) for scale in stage_cfg["final_scales"]]
#     epochs = int(stage_cfg["epochs"])
#     ramp_epochs = int(stage_cfg.get("ramp_epochs", 0))

#     aux_enabled = bool(aux_cfg and aux_cfg.get("enabled", False)) and bool(
#         new_layers
#     )
#     aux_eps = float(aux_cfg.get("eps", 1e-8)) if aux_cfg else 1e-8
#     aux_lambda0 = (
#         float(aux_cfg.get("initial_coeff", 1.0)) if aux_cfg else 0.0
#     )
#     aux_power = float(aux_cfg.get("power", 1.0)) if aux_cfg else 1.0
#     aux_lower_start = min(new_layers) if aux_enabled else None
#     aux_upper_start = max(new_layers) + 1 if aux_enabled else None
#     aux_scalar_layer = aux_upper_start - 1 if aux_enabled else None

#     # A head is attached only when the stage actually stacks a new group; the
#     # first stage is the plain 2-layer base model and must stay comparable to
#     # the non-hyperbolic StackRec runs.
#     hyp_enabled = (
#         bool(hyp_cfg and hyp_cfg.get("enabled", False)) and bool(new_layers)
#     )
#     hyp_head_layer = None
#     hyp_depth = None
#     hyp_curvature = None
#     hyp_loss_coeff = 0.0
#     hyp_warmup = True
#     if hyp_enabled:
#         # The head sits at the output of the group that has just been stacked.
#         hyp_head_layer = max(new_layers)
#         depth_mode = str(hyp_cfg.get("depth_mode", "absolute")).lower()
#         if depth_mode == "absolute":
#             hyp_depth = hyp_head_layer + 1
#         elif depth_mode == "cumulative":
#             hyp_depth = len(active_layers)
#         elif depth_mode == "active_prefix":
#             hyp_depth = sum(1 for idx in active_layers if idx <= hyp_head_layer)
#         else:
#             raise ValueError(f"Unknown hyperbolic depth_mode: {depth_mode}")

#         schedule = str(hyp_cfg.get("schedule", "inverse_power")).lower()
#         reference_curvature = float(hyp_cfg.get("reference_curvature", 1.0))
#         reference_depth = float(hyp_cfg.get("reference_depth", 2.0))
#         curvature_power = float(hyp_cfg.get("power", 2.0))
#         if schedule == "constant":
#             hyp_curvature = reference_curvature
#         elif schedule == "inverse_power":
#             hyp_curvature = reference_curvature * (
#                 reference_depth / max(float(hyp_depth), 1.0)
#             ) ** curvature_power
#         else:
#             raise ValueError(f"Unknown hyperbolic curvature schedule: {schedule}")

#         hyp_loss_coeff = float(hyp_cfg.get("loss_coeff", 0.1))
#         hyp_warmup = bool(hyp_cfg.get("warmup", True))

#     # Lorentz hyperbolic-residual heads.  They are attached to the output of
#     # every newly copied (stacked) group.  Both the residual gate and the
#     # curvature are ramped from exactly zero, so at the moment a group is
#     # stacked the model is still exactly the pretrained subnet.
#     hyp_res_enabled = (
#         bool(hyp_res_cfg and hyp_res_cfg.get("enabled", False))
#         and bool(new_layers)
#     )
#     hyp_res_head_layer = None
#     hyp_res_curvature_target = None
#     hyp_res_residual_ramp = 0
#     hyp_res_curvature_ramp = 0
#     hyp_res_loss_coeff = 0.0
#     hyp_res_warmup = True
#     if hyp_res_enabled:
#         hyp_res_head_layer = max(new_layers)
#         curvature_targets = hyp_res_cfg.get("curvature_targets", {}) or {}
#         hyp_res_curvature_target = float(
#             curvature_targets.get(
#                 stage_name,
#                 stage_cfg.get(
#                     "hyperbolic_curvature_target",
#                     hyp_res_cfg.get("default_curvature_target", 1.0),
#                 ),
#             )
#         )
#         default_ramp = max(int(ramp_epochs), 1)
#         hyp_res_residual_ramp = int(
#             hyp_res_cfg.get("residual_ramp_epochs", default_ramp)
#         )
#         hyp_res_curvature_ramp = int(
#             hyp_res_cfg.get("curvature_ramp_epochs", default_ramp)
#         )
#         hyp_res_loss_coeff = float(hyp_res_cfg.get("loss_coeff", 0.1))
#         hyp_res_warmup = bool(hyp_res_cfg.get("warmup", True))

#     optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
#     if optimizer_type == "muon":
#         lr = None
#         muon_lr = float(stage_cfg["muon_lr"])
#         adamw_lr = float(stage_cfg["adamw_lr"])
#         lr_description = f"muon={muon_lr}, adamw={adamw_lr}"
#     else:
#         lr = float(stage_cfg["lr"])
#         muon_lr = None
#         adamw_lr = lr
#         lr_description = f"adamw={adamw_lr}"

#     logger.info("=" * 80)
#     logger.info(
#         f"STAGE {stage_name}: active_layers={active_layers}, {lr_description}"
#     )
#     logger.info(
#         f"STAGE {stage_name}: epochs={epochs}, ramp_epochs={ramp_epochs}, "
#         f"new_layers={new_layers}, final_scales={final_scales}"
#     )

#     for src_idx, dst_idx in stage_cfg.get("copy_layers", []):
#         src_idx, dst_idx = int(src_idx), int(dst_idx)
#         model.copy_layer_weights(src_idx, dst_idx)
#         logger.info(f"STAGE {stage_name}: copied layer {src_idx} -> layer {dst_idx}")

#     model.set_layers_trainable(range(num_layers), False)
#     model.set_layers_trainable(active_layers, True)

#     if aux_enabled:
#         # The scalar sits on the last layer of the newly stacked lower group,
#         # i.e. exactly at the interface with the pretrained upper subnet.
#         model.enable_layer_aux_scalar(
#             aux_scalar_layer, init=float(aux_cfg.get("scalar_init", 1.0))
#         )
#         model.set_aux_boundary(aux_lower_start, aux_upper_start)
#         logger.info(
#             f"STAGE {stage_name}: aux identity loss on layers "
#             f"[{aux_lower_start}, {aux_upper_start}) "
#             f"(scalar on layer {aux_scalar_layer}), "
#             f"lambda(alpha) = {aux_lambda0:.6g} * (1 - alpha)^{aux_power:g}"
#         )
#     else:
#         model.clear_aux_boundary()

#     if hyp_enabled:
#         if bool(hyp_cfg.get("freeze_previous", True)):
#             model.set_hyperbolic_heads_trainable(None, False)
#         head = model.add_hyperbolic_head(
#             hyp_head_layer,
#             curvature=hyp_curvature,
#             learnable_curvature=bool(
#                 hyp_cfg.get("learnable_curvature", False)
#             ),
#             temperature=float(hyp_cfg.get("temperature", 0.1)),
#             distance=str(hyp_cfg.get("distance", "squared")),
#         )
#         model.set_hyperbolic_heads_trainable([hyp_head_layer], True)
#         logger.info(
#             f"STAGE {stage_name}: hyperbolic head on layer {hyp_head_layer} "
#             f"(depth={hyp_depth}, schedule="
#             f"{hyp_cfg.get('schedule', 'inverse_power')}, "
#             f"c={float(head.curvature().detach().cpu()):.6f}, "
#             f"learnable={bool(hyp_cfg.get('learnable_curvature', False))}), "
#             f"loss_coeff={hyp_loss_coeff:g}"
#         )

#     if hyp_res_enabled:
#         if bool(hyp_res_cfg.get("freeze_previous", True)):
#             model.set_lorentz_heads_trainable(None, False)
#         model.add_lorentz_head(
#             hyp_res_head_layer,
#             curvature=0.0,
#             learnable_curvature=bool(
#                 hyp_res_cfg.get("learnable_curvature", False)
#             ),
#             temperature=float(hyp_res_cfg.get("temperature", 0.1)),
#             distance=str(hyp_res_cfg.get("distance", "squared")),
#         )
#         model.set_lorentz_heads_trainable([hyp_res_head_layer], True)
#         model.set_lorentz_gate(hyp_res_head_layer, 0.0)
#         model.set_lorentz_curvature(hyp_res_head_layer, 0.0)
#         logger.info(
#             f"STAGE {stage_name}: Lorentz residual head on layer "
#             f"{hyp_res_head_layer} (curvature 0 -> "
#             f"{hyp_res_curvature_target:g} over "
#             f"{hyp_res_curvature_ramp} epochs, residual gate 0 -> 1 over "
#             f"{hyp_res_residual_ramp} epochs, "
#             f"loss_coeff={hyp_res_loss_coeff:g})"
#         )

#     # Sanity check: before the ramp starts, newly activated layers contribute
#     # exactly zero, so the model must reproduce the previous stage.
#     stage_start_val_metrics = None
#     stage_start_test_metrics = None
#     if new_layers:
#         start_scales = list(final_scales)
#         for layer_idx in new_layers:
#             start_scales[layer_idx] = 0.0
#         model.set_layer_scales(start_scales)
#         if hyp_res_enabled:
#             model.set_lorentz_gate(hyp_res_head_layer, 0.0)
#             model.set_lorentz_curvature(hyp_res_head_layer, 0.0)
#         stage_start_val_metrics, stage_start_test_metrics = evaluate(
#             val_dataloader, test_dataloader, model, metrics, device
#         )
#         logger.info(
#             f"STAGE {stage_name}: start (new gates=0) "
#             f"val/ndcg@10={stage_start_val_metrics['ndcg@10']:.6f}, "
#             f"test/ndcg@10={stage_start_test_metrics['ndcg@10']:.6f}"
#         )

#     optimizer = make_optimizer(
#         model, adamw_lr=adamw_lr, muon_lr=muon_lr, optimizer_cfg=optimizer_cfg
#     )
#     trainable_params = sum(
#         param.numel() for param in model.parameters() if param.requires_grad
#     )
#     logger.info(f"STAGE {stage_name}: trainable params={trainable_params}")

#     history = []
#     best_complete_val = -float("inf")
#     best_complete_epoch = None
#     best_complete_global_epoch = None
#     best_complete_scales = None
#     best_complete_val_metrics = None
#     best_complete_test_metrics = None
#     best_complete_state = None
#     best_complete_lorentz_gates = None

#     best_any_val = -float("inf")
#     best_any_epoch = None
#     best_any_global_epoch = None
#     best_any_scales = None
#     best_any_val_metrics = None
#     best_any_test_metrics = None

#     final_scales_used = None
#     final_val_metrics = None
#     final_test_metrics = None

#     stage_start_time = time.time()
#     for epoch in range(epochs):
#         scales, complete = stage_scales(stage_cfg, epoch)
#         model.set_layer_scales(scales)
#         final_scales_used = scales

#         alpha = stage_progress(stage_cfg, epoch)
#         aux_coeff = (
#             aux_loss_coefficient(aux_cfg, alpha) if aux_enabled else 0.0
#         )

#         residual_gate = 0.0
#         curvature_fraction = 0.0
#         if hyp_res_enabled:
#             residual_gate = ramp_fraction(epoch, hyp_res_residual_ramp)
#             curvature_fraction = ramp_fraction(epoch, hyp_res_curvature_ramp)
#             model.set_lorentz_gate(hyp_res_head_layer, residual_gate)
#             model.set_lorentz_curvature(
#                 hyp_res_head_layer,
#                 hyp_res_curvature_target * curvature_fraction,
#             )

#         hyp_coeff_epoch = 0.0
#         if hyp_enabled:
#             hyp_coeff_epoch = max(
#                 hyp_coeff_epoch,
#                 hyp_loss_coeff * (alpha if hyp_warmup else 1.0),
#             )
#         if hyp_res_enabled:
#             hyp_coeff_epoch = max(
#                 hyp_coeff_epoch,
#                 hyp_res_loss_coeff
#                 * (residual_gate if hyp_res_warmup else 1.0),
#             )

#         residual_complete = (
#             not hyp_res_enabled
#             or (
#                 residual_gate >= 1.0 - 1.0e-12
#                 and curvature_fraction >= 1.0 - 1.0e-12
#             )
#         )
#         complete = complete and residual_complete

#         train_stats = train_one_epoch(
#             train_dataloader,
#             model,
#             optimizer,
#             loss_function,
#             device,
#             aux_coeff=aux_coeff,
#             aux_eps=aux_eps,
#             hyp_coeff=hyp_coeff_epoch,
#         )
#         val_metrics, test_metrics = evaluate(
#             val_dataloader, test_dataloader, model, metrics, device
#         )

#         global_epoch = global_epoch_start + epoch
#         val_ndcg = float(val_metrics["ndcg@10"])
#         test_ndcg = float(test_metrics["ndcg@10"])
#         layer_aux_scalars = model.get_layer_aux_scalars()

#         history.append(
#             {
#                 "stage": stage_name,
#                 "stage_epoch": epoch,
#                 "global_epoch": global_epoch,
#                 "optimizer_type": optimizer_type,
#                 "lr": lr,
#                 "muon_lr": muon_lr,
#                 "adamw_lr": adamw_lr,
#                 "layer_scales": scales,
#                 "gates_complete": complete,
#                 "aux_enabled": aux_enabled,
#                 "aux_alpha": alpha,
#                 "aux_loss_coeff": aux_coeff,
#                 "layer_aux_scalars": layer_aux_scalars,
#                 "hyperbolic_head": {
#                     "enabled": hyp_enabled,
#                     "head_layer": hyp_head_layer,
#                     "depth": hyp_depth,
#                     "scheduled_curvature": hyp_curvature,
#                     "loss_coeff": hyp_coeff_epoch,
#                     "curvatures": model.get_hyperbolic_curvatures(),
#                 },
#                 "hyperbolic_residual": {
#                     "enabled": hyp_res_enabled,
#                     "head_layer": hyp_res_head_layer,
#                     "curvature_target": hyp_res_curvature_target,
#                     "curvature": (
#                         float(
#                             model._lorentz_heads[
#                                 str(hyp_res_head_layer)
#                             ].curvature().detach().cpu()
#                         )
#                         if hyp_res_enabled
#                         else None
#                     ),
#                     "curvature_fraction": curvature_fraction,
#                     "residual_gate": residual_gate,
#                     "loss_coeff": hyp_coeff_epoch if hyp_res_enabled else 0.0,
#                     "gates": model.get_lorentz_gates(),
#                     "curvatures": model.get_lorentz_curvatures(),
#                 },
#                 "train": train_stats,
#                 "val_metrics": val_metrics,
#                 "test_metrics": test_metrics,
#             }
#         )

#         aux_log = ""
#         if aux_enabled:
#             aux_log = (
#                 f" alpha={alpha:.3f} lambda={aux_coeff:.4f}"
#                 f" aux_loss={train_stats.get('aux_loss', float('nan')):.4f}"
#                 f" ratio={train_stats.get('identity_ratio', float('nan')):.4f}"
#             )
#         hyp_log = ""
#         if hyp_enabled:
#             hyp_log = (
#                 f" hyp_layer={hyp_head_layer}"
#                 f" c={float(model._hyperbolic_heads[str(hyp_head_layer)].curvature().detach().cpu()):.4f}"
#                 f" hyp_coeff={hyp_coeff_epoch:.4f}"
#                 f" hyp_loss={train_stats.get('hyperbolic_loss', float('nan')):.4f}"
#             )
#         hyp_res_log = ""
#         if hyp_res_enabled:
#             hyp_res_log = (
#                 f" res_layer={hyp_res_head_layer}"
#                 f" res_gate={residual_gate:.3f}"
#                 f" res_c={float(model._lorentz_heads[str(hyp_res_head_layer)].curvature().detach().cpu()):.4f}"
#                 f" res_coeff={hyp_coeff_epoch:.4f}"
#                 f" res_loss={train_stats.get('hyperbolic_loss', float('nan')):.4f}"
#             )
#         logger.info(
#             f"STAGE {stage_name} epoch {epoch + 1}/{epochs} "
#             f"scales={[round(scale, 3) for scale in scales]}"
#             f"{aux_log}{hyp_log}{hyp_res_log} "
#             f"val/ndcg@10={val_ndcg:.6f} test/ndcg@10={test_ndcg:.6f}"
#         )

#         if val_ndcg > best_any_val:
#             best_any_val = val_ndcg
#             best_any_epoch = epoch
#             best_any_global_epoch = global_epoch
#             best_any_scales = list(scales)
#             best_any_val_metrics = val_metrics
#             best_any_test_metrics = test_metrics

#         if complete and val_ndcg > best_complete_val:
#             best_complete_val = val_ndcg
#             best_complete_epoch = epoch
#             best_complete_global_epoch = global_epoch
#             best_complete_scales = list(scales)
#             best_complete_val_metrics = val_metrics
#             best_complete_test_metrics = test_metrics
#             best_complete_state = copy.deepcopy(model.state_dict())
#             best_complete_lorentz_gates = model.get_lorentz_gates()

#         final_val_metrics = val_metrics
#         final_test_metrics = test_metrics

#     stage_time = time.time() - stage_start_time

#     if best_complete_state is not None:
#         model.load_state_dict(best_complete_state)
#         model.set_layer_scales(best_complete_scales)
#         # Lorentz curvature buffers are part of the state dict, but the gates
#         # are plain python floats, so restore them explicitly.
#         if best_complete_lorentz_gates is not None:
#             for layer_idx, gate in best_complete_lorentz_gates.items():
#                 if str(int(layer_idx)) in model._lorentz_heads:
#                     model.set_lorentz_gate(int(layer_idx), gate)
#         model.set_layers_trainable(range(num_layers), False)
#         model.set_layers_trainable(active_layers, True)
#         logger.info(
#             f"STAGE {stage_name}: restored best complete checkpoint "
#             f"(stage_epoch={best_complete_epoch}, "
#             f"val/ndcg@10={best_complete_val:.6f})"
#         )
#     else:
#         logger.warning(
#             f"STAGE {stage_name}: no complete-gate epoch found, "
#             "keeping the final weights"
#         )

#     summary = {
#         "name": stage_name,
#         "num_active_layers": len(active_layers),
#         "active_layers": active_layers,
#         "new_layers": new_layers,
#         "optimizer_type": optimizer_type,
#         "lr": lr,
#         "muon_lr": muon_lr,
#         "adamw_lr": adamw_lr,
#         "epochs": epochs,
#         "ramp_epochs": ramp_epochs,
#         "final_scales": final_scales,
#         "copy_layers": stage_cfg.get("copy_layers", []),
#         "trainable_params": trainable_params,
#         "train_time_sec": stage_time,
#         "global_epoch_start": global_epoch_start,
#         "global_epoch_end": global_epoch_start + epochs - 1,
#         "stage_start_val_metrics": stage_start_val_metrics,
#         "stage_start_test_metrics": stage_start_test_metrics,
#         "best_complete_epoch": best_complete_epoch,
#         "best_complete_global_epoch": best_complete_global_epoch,
#         "best_complete_layer_scales": best_complete_scales,
#         "best_complete_val_metrics": best_complete_val_metrics,
#         "best_complete_test_metrics": best_complete_test_metrics,
#         "best_complete_key_metrics": {
#             "val": select_key_metrics(best_complete_val_metrics),
#             "test": select_key_metrics(best_complete_test_metrics),
#         }
#         if best_complete_val_metrics is not None
#         else None,
#         "best_any_epoch": best_any_epoch,
#         "best_any_global_epoch": best_any_global_epoch,
#         "best_any_layer_scales": best_any_scales,
#         "best_any_val_metrics": best_any_val_metrics,
#         "best_any_test_metrics": best_any_test_metrics,
#         "final_layer_scales": final_scales_used,
#         "final_val_metrics": final_val_metrics,
#         "final_test_metrics": final_test_metrics,
#         "final_key_metrics": {
#             "val": select_key_metrics(final_val_metrics),
#             "test": select_key_metrics(final_test_metrics),
#         },
#         "aux_identity": {
#             "enabled": aux_enabled,
#             "lower_start": aux_lower_start,
#             "upper_start": aux_upper_start,
#             "scalar_layer": aux_scalar_layer,
#             "initial_coeff": aux_lambda0,
#             "power": aux_power,
#             "eps": aux_eps,
#             "relation": "lambda(alpha) = lambda0 * (1 - alpha) ** power",
#             "layer_aux_scalars": model.get_layer_aux_scalars(),
#         },
#         "hyperbolic_head": {
#             "enabled": hyp_enabled,
#             "head_layer": hyp_head_layer,
#             "depth": hyp_depth,
#             "depth_mode": (
#                 str(hyp_cfg.get("depth_mode", "absolute")) if hyp_cfg else None
#             ),
#             "schedule": hyp_cfg.get("schedule") if hyp_cfg else None,
#             "scheduled_curvature": hyp_curvature,
#             "loss_coeff": hyp_loss_coeff,
#             "warmup": hyp_warmup,
#             "learnable_curvature": (
#                 bool(hyp_cfg.get("learnable_curvature", False))
#                 if hyp_cfg
#                 else False
#             ),
#             "curvatures": model.get_hyperbolic_curvatures(),
#         },
#         "hyperbolic_residual": {
#             "enabled": hyp_res_enabled,
#             "head_layer": hyp_res_head_layer,
#             "curvature_target": hyp_res_curvature_target,
#             "residual_ramp_epochs": hyp_res_residual_ramp,
#             "curvature_ramp_epochs": hyp_res_curvature_ramp,
#             "loss_coeff": hyp_res_loss_coeff,
#             "warmup": hyp_res_warmup,
#             "normalization": (
#                 "the residual is rescaled to the norm of the boundary state, "
#                 "added with a gate ramped from 0 to 1, and the sum is "
#                 "renormalised back to that norm"
#             ),
#             "gates": model.get_lorentz_gates(),
#             "curvatures": model.get_lorentz_curvatures(),
#         },
#     }

#     return summary, history


# @hydra.main(
#     version_base=None,
#     config_path="configs",
#     config_name="beauty_sasrec_stackrec",
# )
# def main(cfg):
#     output_dir = Path(HydraConfig.get().runtime.output_dir)
#     config = OmegaConf.to_container(cfg, resolve=True)
#     config["model"]["topk_k"] = max(config["metrics_ks"])
#     OmegaConf.save(config, output_dir / "config.yaml")

#     logger.info(f"StackRec config:\n{OmegaConf.to_yaml(config)}\n")

#     fix_random_seed(config.get("seed", 42))
#     device = config["device"]
#     logger.info(f"Current DEVICE: {device}")

#     dataloaders, dataset_meta = prepare_data(config, stage="select_epoch")

#     item2num_interactions = None
#     data_dir = Path(config["dataset"]["path_to_data_dir"])
#     dataset_name = config["dataset"]["name"]
#     train_path = data_dir / "global_split" / dataset_name / "train.csv"
#     if train_path.exists():
#         train_df = pd.read_csv(train_path)
#         item2num_interactions = train_df.groupby("item_id").size().to_dict()

#     model = build_model(config["model_name"], config, dataset_meta).to(device)
#     loss_function = instantiate(config["loss"])
#     metrics = build_metrics(config, dataset_meta, item2num_interactions)

#     logger.info(f"Num params: {sum(p.numel() for p in model.parameters())}")

#     stage_summaries = []
#     histories = {}
#     global_epoch = 0
#     aux_cfg = config["stackrec"].get("aux_identity", {})
#     hyp_cfg = config["stackrec"].get("hyperbolic_heads", {})
#     hyp_res_cfg = config["stackrec"].get("hyperbolic_residual", {})

#     for stage_cfg in config["stackrec"]["stages"]:
#         summary, history = run_stage(
#             stage_cfg=stage_cfg,
#             model=model,
#             train_dataloader=dataloaders["train"],
#             val_dataloader=dataloaders["validation"],
#             test_dataloader=dataloaders["test"],
#             loss_function=loss_function,
#             metrics=metrics,
#             device=device,
#             optimizer_cfg=config["optimizer"],
#             global_epoch_start=global_epoch,
#             aux_cfg=aux_cfg,
#             hyp_cfg=hyp_cfg,
#             hyp_res_cfg=hyp_res_cfg,
#         )
#         stage_summaries.append(summary)
#         histories[summary["name"]] = history
#         global_epoch += int(stage_cfg["epochs"])

#         with open(output_dir / f"{summary['name']}_best_metrics.json", "w") as f:
#             json.dump(
#                 {
#                     "best_complete": {
#                         "val": summary["best_complete_val_metrics"],
#                         "test": summary["best_complete_test_metrics"],
#                     },
#                     "final": {
#                         "val": summary["final_val_metrics"],
#                         "test": summary["final_test_metrics"],
#                     },
#                 },
#                 f,
#                 indent=2,
#             )

#     results = {
#         "description": (
#             f"{config['optimizer'].get('type', 'adamw')} + StackRec on "
#             "SASRec/Beauty. One continuous training run that progressively "
#             "activates the last 2, middle 4 and all 6 transformer layers."
#         ),
#         "optimizer_type": config["optimizer"].get("type", "adamw"),
#         "interpretation": {
#             "layer_scales": (
#                 "Per-layer forward coefficient. A zero coefficient makes the "
#                 "layer an identity (it is skipped); newly activated layers are "
#                 "ramped from 0 to 1."
#             ),
#             "stages": {
#                 "2_layers": "layers 4,5 trained; layers 0-3 frozen with scale 0",
#                 "4_layers": "layer 4->2 and 5->3; layers 2-5 trained; scales of 2,3 ramped",
#                 "6_layers": "layer 2->0 and 3->1; all 6 layers trained; scales of 0,1 ramped",
#             },
#             "aux_identity": (
#                 "When a new group is stacked below the trained subnet, the "
#                 "output scalar of its last layer becomes trainable and an "
#                 "auxiliary loss drives ||g * output|| / ||input|| to 1 at the "
#                 "interface. Its weight follows the linear-complement law "
#                 "lambda(alpha) = lambda0 * (1 - alpha), so it decays linearly "
#                 "while the new-group scale alpha ramps from 0 to 1."
#             ),
#             "selection": (
#                 "best_complete_* are metrics at the best validation ndcg@10 "
#                 "among epochs where all ramp coefficients reached their final "
#                 "values. final_* are metrics at the last epoch of the stage."
#             ),
#             "hyperbolic_heads": (
#                 "Optional Poincare-ball next-item heads attached to the output "
#                 "of each newly stacked group. The boundary representation and "
#                 "the item table are mapped through expmap0 and scored by "
#                 "negative hyperbolic distance. Curvature follows "
#                 "c(depth) = c_ref * (depth_ref / depth) ** power (or is "
#                 "learned), so deeper/upper heads use a flatter, larger ball."
#             ),
#             "hyperbolic_residual": (
#                 "Lorentz/hyperboloid residual heads attached to the output of "
#                 "every copied group. The residual is the curvature-induced "
#                 "correction of the Lorentz exponential map, so it is exactly "
#                 "zero at c=0. It is multiplied by a gate ramped from 0 to 1, "
#                 "rescaled to the norm of the boundary state, added, and the "
#                 "sum is renormalised back to that norm. Thus at the moment a "
#                 "group is stacked the model is exactly the pretrained subnet, "
#                 "and the input norm of the pretrained subnet is preserved "
#                 "throughout the ramp. Curvature ramps from 0 to the stage "
#                 "target (0.2 for the first copy, 1.0 for the second copy)."
#             ),
#         },
#         "config": config,
#         "stages": {summary["name"]: summary for summary in stage_summaries},
#         "summary": {
#             summary["name"]: {
#                 "num_active_layers": summary["num_active_layers"],
#                 "optimizer_type": summary["optimizer_type"],
#                 "lr": summary["lr"],
#                 "muon_lr": summary["muon_lr"],
#                 "adamw_lr": summary["adamw_lr"],
#                 "epochs": summary["epochs"],
#                 "best_complete_epoch": summary["best_complete_epoch"],
#                 "best_complete_global_epoch": summary[
#                     "best_complete_global_epoch"
#                 ],
#                 "best_complete": summary["best_complete_key_metrics"],
#                 "final": summary["final_key_metrics"],
#                 "aux_identity": summary["aux_identity"],
#                 "hyperbolic_head": summary["hyperbolic_head"],
#                 "hyperbolic_residual": summary["hyperbolic_residual"],
#             }
#             for summary in stage_summaries
#         },
#     }

#     with open(output_dir / "stackrec_results.json", "w") as f:
#         json.dump(results, f, indent=2)
#     with open(output_dir / config["stackrec"]["history_path"], "w") as f:
#         json.dump(histories, f, indent=2)

#     results_path = Path(config["stackrec"]["results_path"])
#     with open(results_path, "w") as f:
#         json.dump(results, f, indent=2)

#     logger.info(f"Saved stage results to {output_dir / 'stackrec_results.json'}")
#     logger.info(f"Saved history to {output_dir / config['stackrec']['history_path']}")
#     logger.info(f"Saved summary to {results_path}")

#     for summary in stage_summaries:
#         logger.info(
#             f"RESULT {summary['name']}: "
#             f"val/ndcg@10={summary['best_complete_val_metrics']['ndcg@10']:.6f}, "
#             f"test/ndcg@10={summary['best_complete_test_metrics']['ndcg@10']:.6f}, "
#             f"val/recall@10={summary['best_complete_val_metrics']['recall@10']:.6f}, "
#             f"test/recall@10={summary['best_complete_test_metrics']['recall@10']:.6f}"
#         )


# if __name__ == "__main__":
#     main()
