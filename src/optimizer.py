import copy
import math

import torch

OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "cyclic": torch.optim.lr_scheduler.CyclicLR,
}


def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    """Orthogonalize a 2D matrix with the Newton-Schulz iteration.

    Follows Keller Jordan's Muon implementation. The iteration computes
    approximately U @ V^T for the SVD g = U @ S @ V^T, i.e. all singular
    values are pushed towards one.
    """
    assert g.ndim == 2, "Newton-Schulz orthogonalization expects a 2D tensor"

    a, b, c = 3.4445, -4.7750, 2.0315

    x = g.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.t()

    x = x / (x.norm() + eps)

    for _ in range(steps):
        mat = x @ x.t()
        mat = b * mat + c * (mat @ mat)
        x = a * x + mat @ x

    if transpose:
        x = x.t()

    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D (matrix) parameters.

    The momentum-smoothed gradient is orthogonalized with Newton-Schulz and
    scaled by sqrt(max(1, rows / cols)), which keeps the update RMS roughly
    comparable to AdamW across matrix shapes.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                assert grad.ndim == 2, "Muon only supports 2D parameters"

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(buffer, momentum) if nesterov else buffer

                update = zeropower_via_newtonschulz5(grad, steps=ns_steps)
                update *= max(1, update.size(0) / update.size(1)) ** 0.5

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                param.add_(update, alpha=-lr)

        return loss


class MuonAdamW:
    """Mixed optimizer: Muon for 2D weights, AdamW for everything else."""

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.muon.step(closure)
        self.adamw.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def _get_lr_scale(config, num_layers):
    """Depth-dependent LR scale: lr(L) = lr_ref * sqrt(L_ref / L)."""
    scaling_cfg = config.get("lr_scaling")
    if scaling_cfg is None:
        return 1.0

    scaling_type = scaling_cfg.get("type", "sqrt_depth")
    if scaling_type != "sqrt_depth":
        raise ValueError(f"Unknown lr_scaling type: {scaling_type}")

    reference_num_layers = scaling_cfg.get("reference_num_layers", num_layers)
    return math.sqrt(reference_num_layers / num_layers)


def _create_muon_adamw(optimizer_cfg, model, lr_scale):
    lr = optimizer_cfg.pop("lr") * lr_scale
    adamw_lr = optimizer_cfg.pop("adamw_lr") * lr_scale
    momentum = optimizer_cfg.pop("momentum", 0.95)
    nesterov = optimizer_cfg.pop("nesterov", True)
    ns_steps = optimizer_cfg.pop("ns_steps", 5)
    weight_decay = optimizer_cfg.pop("weight_decay", 0.0)
    adamw_weight_decay = optimizer_cfg.pop("adamw_weight_decay", 0.0)
    adamw_betas = tuple(optimizer_cfg.pop("adamw_betas", (0.9, 0.999)))
    adamw_eps = optimizer_cfg.pop("adamw_eps", 1e-8)
    exclude_embeddings = optimizer_cfg.pop("exclude_embeddings", False)

    if optimizer_cfg:
        raise ValueError(f"Unknown Muon optimizer args: {sorted(optimizer_cfg)}")

    muon_params = []
    adamw_params = []
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
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
    )
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=adamw_betas,
        eps=adamw_eps,
        weight_decay=adamw_weight_decay,
    )

    return MuonAdamW(muon=muon, adamw=adamw)


class BasicOptimizer:
    def __init__(self, model, optimizer, scheduler=None, clip_grad_threshold=None):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._clip_grad_threshold = clip_grad_threshold

    @classmethod
    def create_from_config(cls, config, **kwargs):
        optimizer_cfg = copy.deepcopy(config["optimizer"])
        optimizer_type = optimizer_cfg.pop("type")
        lr_scale = _get_lr_scale(config, kwargs.get("num_layers"))

        if optimizer_type == "muon":
            optimizer = _create_muon_adamw(
                optimizer_cfg, kwargs["model"], lr_scale
            )
        else:
            optimizer_cfg["lr"] *= lr_scale
            optimizer = OPTIMIZERS[optimizer_type](
                kwargs["model"].parameters(), **optimizer_cfg
            )

        if "scheduler" in config:
            scheduler_cfg = copy.deepcopy(config["scheduler"])
            scheduler = SCHEDULERS[scheduler_cfg.pop("type")](
                optimizer, **scheduler_cfg
            )
        else:
            scheduler = None

        return cls(
            model=kwargs["model"],
            optimizer=optimizer,
            scheduler=scheduler,
            clip_grad_threshold=config.get("clip_grad_threshold", None),
        )

    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_grad_threshold
            )

        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

    def state_dict(self):
        state_dict = {"optimizer": self._optimizer.state_dict()}
        if self._scheduler is not None:
            state_dict.update({"scheduler": self._scheduler.state_dict()})
        return state_dict
