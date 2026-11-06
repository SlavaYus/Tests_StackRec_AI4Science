import math

import torch
from torch import nn

from ..utils import create_masked_tensor


# ============================================================
# Lorentz-операции
# ============================================================
# Соглашение:
#   Minkowski: <x,y>_L = -x_0 y_0 + sum_{i>=1} x_i y_i
#   Гиперболоид H^n_K = {x : <x,x>_L = -K, x_0 > 0}, K > 0
#   Кривизна многообразия = -K.
# ============================================================


def exp_map_origin(v, K):
    """Экспоненциальное отображение из T_0 H^n_K в H^n_K.

    v: (..., D)  — касательный вектор в начале координат
    K: (скаляр или (...,))  — положительная кривизна
    Возвращает: (..., D+1) точку на гиперболоиде.
    """
    sqrt_K = K.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x_0 = sqrt_K * torch.cosh(v_norm / sqrt_K)
    x_sp = sqrt_K * torch.sinh(v_norm / sqrt_K) * v / v_norm
    return torch.cat([x_0, x_sp], dim=-1)


def log_map_origin(x, K):
    """Обратное отображение: H^n_K -> T_0 H^n_K."""
    sqrt_K = K.sqrt()
    x_0 = x[..., :1]
    x_sp = x[..., 1:]
    x_sp_norm = x_sp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # arcosh(x_0 / sqrt(K)); clamp для численной стабильности
    arg = (x_0 / sqrt_K).clamp_min(1.0 + 1e-7)
    v_norm = sqrt_K * torch.acosh(arg)
    return v_norm * x_sp / x_sp_norm


def project_to_hyperboloid(x, K):
    """Проекция точки Минковского на гиперболоид H^n_K."""
    # <x,x>_L = -x_0^2 + ||x_sp||^2
    mink_sq = -(x[..., :1] ** 2) + (x[..., 1:] ** 2).sum(-1, keepdim=True)
    # Нужно <x,x>_L = -K  =>  масштаб = sqrt(K / -<x,x>_L)
    scale = (K / (-mink_sq).clamp_min(1e-8)).sqrt()
    return x * scale


# ============================================================
# Голова с лоренцевым residual
# ============================================================


class LorentzResidualHead(nn.Module):
    """Residual-голова на гиперболоиде Лоренца.

    Схема:
      1. h и f(h) отображаются на гиперболоид через exp_0.
      2. Считается взвешенный центр тяжести в объемлющем пространстве
         Минковского:  h_raw = w_x * exp_0(h) + w_y * exp_0(f(h)).
      3. h_raw проецируется обратно на гиперболоид.
      4. Через log_0 результат переводится в касательное пространство.
      5. Residual = log_0(projected) - h.
    """

    def __init__(
        self,
        embedding_dim,
        curvature=1.0,
        learnable_curvature=True,
        init_weight=0.5,
    ):
        super().__init__()
        # Линейное "предложение" f(h) в касательном пространстве.
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.proj.weight)  # стартуем как identity

        # Обучаемая кривизна: храним log K для положительности.
        init_log_k = math.log(float(curvature))
        if learnable_curvature:
            self.log_curvature = nn.Parameter(torch.tensor(init_log_k))
        else:
            self.register_buffer("log_curvature", torch.tensor(init_log_k))

        # Обучаемый вес смешивания (сигмоида от скаляра).
        # w_x = sigmoid(w), w_y = 1 - w_x.
        self._w_logit = nn.Parameter(torch.tensor(
            math.log(init_weight / (1.0 - init_weight))
        ))

    # ---------- свойства ----------

    def curvature(self):
        """Возвращает положительную K (кривизна многообразия = -K)."""
        return self.log_curvature.exp()

    def set_curvature(self, value):
        with torch.no_grad():
            self.log_curvature.fill_(math.log(max(float(value), 1e-6)))

    def mixing_weights(self):
        w_x = torch.sigmoid(self._w_logit)
        return w_x, 1.0 - w_x

    # ---------- residual ----------

    def residual(self, h):
        """h: (..., D) — касательный вектор (скрытое состояние).

        Возвращает residual той же формы в касательном пространстве.
        """
        K = self.curvature()
        f_h = self.proj(h)

        x_M = exp_map_origin(h, K)
        y_M = exp_map_origin(f_h, K)

        w_x, w_y = self.mixing_weights()
        h_raw = w_x * x_M + w_y * y_M

        h_on_manifold = project_to_hyperboloid(h_raw, K)
        h_new = log_map_origin(h_on_manifold, K)
        return h_new - h


# ============================================================
# Расписание кривизны по глубине
# ============================================================


class CurvatureSchedule(nn.Module):
    """Параметрическое расписание K_ell как функции номера слоя.

    Режимы:
      * 'independent' — K_l — независимые обучаемые скаляры;
      * 'exponential' — K_l = K_0 * gamma^l, gamma обучаемая в (0,1);
      * 'logistic'    — K_l = (K_max - K_min)/(1+e^{-k(l-l0)}) + K_min.

    В режиме 'independent' метод `curvatures()` возвращает список
    скаляров длины num_layers; в остальных — тоже список, чтобы
    интерфейс был единым.
    """

    def __init__(self, num_layers, mode="independent",
                 init_K=1.0, init_gamma=0.9,
                 K_min=0.1, K_max=2.0, init_slope=1.0, init_mid=None):
        super().__init__()
        self.num_layers = int(num_layers)
        self.mode = mode

        if mode == "independent":
            self.log_K = nn.Parameter(
                torch.full((num_layers,), math.log(init_K))
            )
        elif mode == "exponential":
            self.log_K0 = nn.Parameter(torch.tensor(math.log(init_K)))
            # gamma храним как logit, чтобы быть в (0,1)
            self._gamma_logit = nn.Parameter(torch.tensor(
                math.log(init_gamma / (1.0 - init_gamma))
            ))
        elif mode == "logistic":
            self.log_K_min = nn.Parameter(torch.tensor(math.log(K_min)))
            self.log_K_max = nn.Parameter(torch.tensor(math.log(K_max)))
            self._slope = nn.Parameter(torch.tensor(float(init_slope)))
            mid = init_mid if init_mid is not None else num_layers / 2.0
            self._mid = nn.Parameter(torch.tensor(float(mid)))
        else:
            raise ValueError(f"Unknown curvature mode: {mode}")

    def curvatures(self):
        if self.mode == "independent":
            return self.log_K.exp()  # (num_layers,)
        if self.mode == "exponential":
            l = torch.arange(self.num_layers,
                             device=self.log_K0.device, dtype=self.log_K0.dtype)
            gamma = torch.sigmoid(self._gamma_logit)
            return self.log_K0.exp() * gamma.pow(l)
        # logistic
        l = torch.arange(self.num_layers,
                         device=self.log_K_min.device, dtype=self.log_K_min.dtype)
        K_min = self.log_K_min.exp()
        K_max = self.log_K_max.exp()
        return (K_max - K_min) / (1.0 + torch.exp(
            -self._slope * (l - self._mid)
        )) + K_min


# ============================================================
# Энкодер
# ============================================================


class SequentialEncoder(nn.Module):
    """StackRec + Lorentz residual + обучаемая кривизна по слоям.

    Отличия от евклидовой версии:
      * на границе каждого нового блока можно включить Lorentz-residual;
      * кривизна задаётся на каждом слое через CurvatureSchedule;
      * residual гейтируется, чтобы при stacking новая группа стартовала
        как identity (gate == 0), и плавно вводилась в работу.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dim_feedforward,
        dropout,
        activation,
        layer_norm_eps,
        num_layers,
        num_items,
        topk_k,
        position_embeddings,
        item_embeddings,
        # --- гиперболические параметры ---
        curvature_mode="independent",
        init_curvature=1.0,
        init_gamma=0.9,
        K_min=0.1,
        K_max=2.0,
        learnable_curvature=True,
    ):
        super().__init__()
        self._topk_k = topk_k
        self._num_items = num_items

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)
        self._embedding_dim = embedding_dim
        self._position_embeddings = position_embeddings
        self._item_embeddings = item_embeddings
        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        # StackRec: forward-коэффициенты
        self.register_buffer("_layer_scales", torch.ones(num_layers), persistent=False)
        self._layer_scale_values = [1.0] * num_layers

        # Aux scalars
        self._layer_aux_scalars = nn.ParameterList(
            [nn.Parameter(torch.ones(())) for _ in range(num_layers)]
        )
        for s in self._layer_aux_scalars.parameters():
            s.requires_grad_(False)

        self._aux_active = False
        self._aux_lower_start = None
        self._aux_upper_start = None

        # --- гиперболическая часть ---
        self._lorentz_heads = nn.ModuleDict()
        self._lorentz_gates = {}  # layer_idx -> python float

        # Расписание кривизны — одно на модель, индексируется по слою.
        self._curvature_schedule = CurvatureSchedule(
            num_layers=num_layers,
            mode=curvature_mode,
            init_K=init_curvature,
            init_gamma=init_gamma,
            K_min=K_min, K_max=K_max,
        )
        if not learnable_curvature:
            for p in self._curvature_schedule.parameters():
                p.requires_grad_(False)

        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    # ---------- свойства / утилиты ----------

    @property
    def num_layers(self):
        return len(self._encoder.layers)

    @property
    def device(self):
        return self._device_anchor.device

    def set_layer_scales(self, scales):
        scales = [float(s) for s in scales]
        if len(scales) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} scales, got {len(scales)}")
        if any(s < 0 for s in scales):
            raise ValueError(f"Layer scales must be non-negative, got {scales}")
        self._layer_scale_values = scales
        with torch.no_grad():
            self._layer_scales.copy_(
                torch.as_tensor(scales, dtype=self._layer_scales.dtype,
                                device=self._layer_scales.device)
            )

    def get_layer_scales(self):
        return list(self._layer_scale_values)

    def set_layers_trainable(self, layer_indices, trainable=True):
        for idx in layer_indices:
            for p in self._encoder.layers[idx].parameters():
                p.requires_grad = trainable

    def copy_layer_weights(self, src_idx, dst_idx):
        if src_idx == dst_idx:
            return
        self._encoder.layers[dst_idx].load_state_dict(
            self._encoder.layers[src_idx].state_dict()
        )

    def enable_layer_aux_scalar(self, layer_idx, init=1.0):
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer index out of range: {layer_idx}")
        with torch.no_grad():
            self._layer_aux_scalars[layer_idx].fill_(float(init))
        self._layer_aux_scalars[layer_idx].requires_grad_(True)

    def set_aux_boundary(self, lower_start, upper_start):
        lower_start, upper_start = int(lower_start), int(upper_start)
        if not 0 <= lower_start < upper_start <= self.num_layers:
            raise ValueError(
                f"Bad boundary: 0 <= {lower_start} < {upper_start} <= {self.num_layers}"
            )
        self._aux_lower_start = lower_start
        self._aux_upper_start = upper_start
        self._aux_active = True

    def clear_aux_boundary(self):
        self._aux_active = False
        self._aux_lower_start = None
        self._aux_upper_start = None

    def get_layer_aux_scalars(self):
        return [float(s.detach().cpu()) for s in self._layer_aux_scalars]

    # ---------- Lorentz-головы ----------

    def add_lorentz_head(self, layer_idx, curvature=None,
                         learnable_curvature=True, init_weight=0.5):
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer index out of range: {layer_idx}")
        key = str(layer_idx)
        if key in self._lorentz_heads:
            return self._lorentz_heads[key]

        if curvature is None:
            curvature = float(self._curvature_schedule.curvatures()[layer_idx])
        head = LorentzResidualHead(
            embedding_dim=self._embedding_dim,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            init_weight=init_weight,
        )
        head.to(self.device)
        self._lorentz_heads[key] = head
        self._lorentz_gates[layer_idx] = 0.0
        return head

    def set_lorentz_heads_trainable(self, layer_indices=None, trainable=True):
        keys = (list(self._lorentz_heads.keys()) if layer_indices is None
                else [str(int(i)) for i in layer_indices])
        for key in keys:
            if key in self._lorentz_heads:
                for p in self._lorentz_heads[key].parameters():
                    p.requires_grad = bool(trainable)

    def set_lorentz_gate(self, layer_idx, gate):
        layer_idx = int(layer_idx)
        if str(layer_idx) not in self._lorentz_heads:
            raise ValueError(f"No Lorentz head at layer {layer_idx}")
        self._lorentz_gates[layer_idx] = max(0.0, float(gate))

    def get_lorentz_gates(self):
        return dict(self._lorentz_gates)

    def get_curvatures(self):
        return self._curvature_schedule.curvatures().detach().cpu().tolist()

    def set_curvature(self, layer_idx, value):
        """Ручная установка кривизны для режима 'independent'."""
        if self._curvature_schedule.mode != "independent":
            raise RuntimeError("Manual set_curvature requires mode='independent'")
        with torch.no_grad():
            self._curvature_schedule.log_K[int(layer_idx)].fill_(
                math.log(max(float(value), 1e-6))
            )

    # ---------- применение residual ----------

    def _apply_lorentz_residual(self, hidden_states, layer_idx):
        """Гейт + norm-matching + добавление residual.

        При gate == 0 возвращает вход без изменений (bitwise),
        поэтому stacking новой группы точно воспроизводит
        предобученную подсеть.
        """
        head = self._lorentz_heads[str(layer_idx)]
        gate = float(self._lorentz_gates.get(int(layer_idx), 0.0))
        if gate == 0.0:
            return hidden_states

        eps = 1e-8
        residual = head.residual(hidden_states)
        norm_in = hidden_states.norm(dim=-1, keepdim=True)
        valid = norm_in > eps
        safe_norm_in = norm_in.clamp_min(eps)

        # norm-match residual к входу
        norm_res = residual.norm(dim=-1, keepdim=True).clamp_min(eps)
        residual = residual * (safe_norm_in / norm_res)

        combined = hidden_states + gate * residual
        # renormalise обратно к исходной норме
        norm_out = combined.norm(dim=-1, keepdim=True).clamp_min(eps)
        combined = combined * (safe_norm_in / norm_out)
        return torch.where(valid, combined, hidden_states)

    # ---------- прямой проход ----------

    def _prepare_sequence(self, seq_len, mask, batch_size,
                          all_sample_lengths, sequence_embeddings):
        positions = (
            torch.arange(seq_len - 1, -1, -1, device=mask.device)[None]
            .tile([batch_size, 1]).long()
        )
        positions_mask = positions < all_sample_lengths[:, None]
        positions = positions[positions_mask]
        pos_emb, _ = create_masked_tensor(
            data=self._position_embeddings(positions),
            lengths=all_sample_lengths,
            device=self.device,
        )
        sequence_embeddings = sequence_embeddings + pos_emb
        sequence_embeddings = self._layernorm(sequence_embeddings)
        sequence_embeddings = self._dropout(sequence_embeddings)
        sequence_embeddings[~mask] = 0
        return sequence_embeddings

    def _encode_sequence(self, seq_len, batch_size, mask, sequence_embeddings):
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=self.device)
        ).bool()

        aux_input = None
        aux_output = None

        for layer_idx, layer in enumerate(self._encoder.layers):
            if self._aux_active and layer_idx == self._aux_lower_start:
                aux_input = sequence_embeddings

            scale = self._layer_scale_values[layer_idx]
            if scale != 0.0:
                sequence_embeddings = layer(
                    sequence_embeddings,
                    src_mask=~causal_mask,
                    src_key_padding_mask=~mask,
                )
                if scale != 1.0:
                    sequence_embeddings = sequence_embeddings * scale
                sequence_embeddings = (
                    sequence_embeddings * self._layer_aux_scalars[layer_idx]
                )
                if str(layer_idx) in self._lorentz_heads:
                    sequence_embeddings = self._apply_lorentz_residual(
                        sequence_embeddings, layer_idx
                    )

            if self._aux_active and layer_idx == self._aux_upper_start - 1:
                aux_output = sequence_embeddings

        if self._encoder.norm is not None:
            sequence_embeddings = self._encoder.norm(sequence_embeddings)
        return sequence_embeddings, aux_input, aux_output

    @staticmethod
    def _get_last_embedding(embeddings, mask):
        lengths = mask.sum(dim=-1) - 1
        last_masks = mask.gather(dim=1, index=lengths[:, None])
        idx = lengths[:, None, None].expand(-1, 1, embeddings.shape[-1])
        last = embeddings.gather(dim=1, index=idx)[last_masks]
        return last

    def forward(self, inputs):
        all_sample_lengths = inputs["item.length"]
        padded_sequence = inputs["padded_sequence_ids"]
        mask = inputs["mask"]
        batch_size, seq_len = mask.shape

        sequence_embeddings = self._item_embeddings(padded_sequence)
        sequence_embeddings *= self._item_embeddings.embedding_dim ** 0.5
        sequence_embeddings = self._prepare_sequence(
            seq_len, mask, batch_size, all_sample_lengths, sequence_embeddings
        )

        sequence_embeddings, aux_input, aux_output = self._encode_sequence(
            seq_len, batch_size, mask, sequence_embeddings
        )

        all_sample_sequence_embeddings = sequence_embeddings[mask]
        sequence_scores = (
            all_sample_sequence_embeddings @ self._item_embeddings.weight.T
        )

        if self.training:
            outputs = {"local_prediction": sequence_scores}
            if aux_input is not None and aux_output is not None:
                outputs["aux_input"] = aux_input
                outputs["aux_output"] = aux_output
            return outputs

        last_embeddings = self._get_last_embedding(sequence_embeddings, mask)
        candidate_scores = last_embeddings @ self._item_embeddings.weight.T
        candidate_scores[:, 0] = -torch.inf
        candidate_scores[:, self._num_items + 1:] = -torch.inf
        _, indices = torch.topk(candidate_scores, k=self._topk_k, dim=-1, largest=True)
        return indices


# import torch
# from torch import nn

# from ..utils import create_masked_tensor


# class SequentialEncoder(nn.Module):
#     """StackRec encoder без какой-либо гиперболической геометрии.

#     Всё, что связано с Poincaré/Lorentz, убрано. Осталась только
#     StackRec-механика:
#       * per-layer forward scales (обнуление = пропуск слоя);
#       * per-layer aux scalars для identity-aux-loss;
#       * отслеживание границы между новым нижним и старым верхним блоком.
#     """

#     def __init__(
#         self,
#         embedding_dim,
#         num_heads,
#         dim_feedforward,
#         dropout,
#         activation,
#         layer_norm_eps,
#         num_layers,
#         num_items,
#         topk_k,
#         position_embeddings,
#         item_embeddings,
#     ):
#         super().__init__()
#         self._topk_k = topk_k
#         self._num_items = num_items

#         transformer_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embedding_dim,
#             nhead=num_heads,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             layer_norm_eps=layer_norm_eps,
#             batch_first=True,
#         )
#         self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)
#         self._embedding_dim = embedding_dim
#         self._position_embeddings = position_embeddings
#         self._item_embeddings = item_embeddings
#         self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
#         self._dropout = nn.Dropout(dropout)

#         # StackRec: forward-коэффициенты по слоям.
#         self.register_buffer("_layer_scales", torch.ones(num_layers), persistent=False)
#         self._layer_scale_values = [1.0] * num_layers

#         # Aux scalars для identity-aux-loss.
#         self._layer_aux_scalars = nn.ParameterList(
#             [nn.Parameter(torch.ones(())) for _ in range(num_layers)]
#         )
#         for scalar in self._layer_aux_scalars.parameters():
#             scalar.requires_grad_(False)

#         self._aux_active = False
#         self._aux_lower_start = None
#         self._aux_upper_start = None

#         self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

#     # ---------- базовые свойства / утилиты ----------

#     @property
#     def num_layers(self):
#         return len(self._encoder.layers)

#     @property
#     def device(self):
#         return self._device_anchor.device

#     def set_layer_scales(self, scales):
#         scales = [float(s) for s in scales]
#         if len(scales) != self.num_layers:
#             raise ValueError(f"Expected {self.num_layers} scales, got {len(scales)}")
#         if any(s < 0 for s in scales):
#             raise ValueError(f"Layer scales must be non-negative, got {scales}")
#         self._layer_scale_values = scales
#         with torch.no_grad():
#             self._layer_scales.copy_(
#                 torch.as_tensor(scales, dtype=self._layer_scales.dtype,
#                                 device=self._layer_scales.device)
#             )

#     def get_layer_scales(self):
#         return list(self._layer_scale_values)

#     def set_layers_trainable(self, layer_indices, trainable=True):
#         for idx in layer_indices:
#             for p in self._encoder.layers[idx].parameters():
#                 p.requires_grad = trainable

#     def copy_layer_weights(self, src_idx, dst_idx):
#         if src_idx == dst_idx:
#             return
#         self._encoder.layers[dst_idx].load_state_dict(
#             self._encoder.layers[src_idx].state_dict()
#         )

#     def enable_layer_aux_scalar(self, layer_idx, init=1.0):
#         layer_idx = int(layer_idx)
#         if not 0 <= layer_idx < self.num_layers:
#             raise ValueError(f"layer index out of range: {layer_idx}")
#         with torch.no_grad():
#             self._layer_aux_scalars[layer_idx].fill_(float(init))
#         self._layer_aux_scalars[layer_idx].requires_grad_(True)

#     def set_aux_boundary(self, lower_start, upper_start):
#         lower_start, upper_start = int(lower_start), int(upper_start)
#         if not 0 <= lower_start < upper_start <= self.num_layers:
#             raise ValueError(
#                 f"Bad boundary: 0 <= {lower_start} < {upper_start} <= {self.num_layers}"
#             )
#         self._aux_lower_start = lower_start
#         self._aux_upper_start = upper_start
#         self._aux_active = True

#     def clear_aux_boundary(self):
#         self._aux_active = False
#         self._aux_lower_start = None
#         self._aux_upper_start = None

#     def get_layer_aux_scalars(self):
#         return [float(s.detach().cpu()) for s in self._layer_aux_scalars]

#     # ---------- прямой проход ----------

#     def _prepare_sequence(self, seq_len, mask, batch_size,
#                           all_sample_lengths, sequence_embeddings):
#         positions = (
#             torch.arange(seq_len - 1, -1, -1, device=mask.device)[None]
#             .tile([batch_size, 1]).long()
#         )
#         positions_mask = positions < all_sample_lengths[:, None]
#         positions = positions[positions_mask]
#         pos_emb, _ = create_masked_tensor(
#             data=self._position_embeddings(positions),
#             lengths=all_sample_lengths,
#             device=self.device,
#         )
#         sequence_embeddings = sequence_embeddings + pos_emb
#         sequence_embeddings = self._layernorm(sequence_embeddings)
#         sequence_embeddings = self._dropout(sequence_embeddings)
#         sequence_embeddings[~mask] = 0
#         return sequence_embeddings

#     def _encode_sequence(self, seq_len, batch_size, mask,
#                          sequence_embeddings, account_user_embedding=False):
#         causal_mask = torch.tril(
#             torch.ones(seq_len, seq_len, device=self.device)
#         ).bool()
#         advanced_mask = causal_mask
#         advanced_src_key_padding_mask = mask

#         aux_input = None
#         aux_output = None

#         for layer_idx, layer in enumerate(self._encoder.layers):
#             if self._aux_active and layer_idx == self._aux_lower_start:
#                 aux_input = sequence_embeddings

#             scale = self._layer_scale_values[layer_idx]
#             if scale != 0.0:
#                 sequence_embeddings = layer(
#                     sequence_embeddings,
#                     src_mask=~advanced_mask,
#                     src_key_padding_mask=~advanced_src_key_padding_mask,
#                 )
#                 if scale != 1.0:
#                     sequence_embeddings = sequence_embeddings * scale
#                 sequence_embeddings = (
#                     sequence_embeddings * self._layer_aux_scalars[layer_idx]
#                 )

#             if self._aux_active and layer_idx == self._aux_upper_start - 1:
#                 aux_output = sequence_embeddings

#         if self._encoder.norm is not None:
#             sequence_embeddings = self._encoder.norm(sequence_embeddings)

#         return sequence_embeddings, aux_input, aux_output

#     @staticmethod
#     def _get_last_embedding(embeddings, mask):
#         lengths = mask.sum(dim=-1) - 1
#         last_masks = mask.gather(dim=1, index=lengths[:, None])
#         idx = lengths[:, None, None].expand(-1, 1, embeddings.shape[-1])
#         last = embeddings.gather(dim=1, index=idx)[last_masks]
#         return last

#     def forward(self, inputs):
#         all_sample_lengths = inputs["item.length"]
#         padded_sequence = inputs["padded_sequence_ids"]
#         mask = inputs["mask"]
#         batch_size, seq_len = mask.shape

#         sequence_embeddings = self._item_embeddings(padded_sequence)
#         sequence_embeddings *= self._item_embeddings.embedding_dim ** 0.5
#         sequence_embeddings = self._prepare_sequence(
#             seq_len, mask, batch_size, all_sample_lengths, sequence_embeddings
#         )

#         sequence_embeddings, aux_input, aux_output = self._encode_sequence(
#             seq_len, batch_size, mask, sequence_embeddings
#         )

#         all_sample_sequence_embeddings = sequence_embeddings[mask]
#         sequence_scores = (
#             all_sample_sequence_embeddings @ self._item_embeddings.weight.T
#         )

#         if self.training:
#             outputs = {"local_prediction": sequence_scores}
#             if aux_input is not None and aux_output is not None:
#                 outputs["aux_input"] = aux_input
#                 outputs["aux_output"] = aux_output
#             return outputs

#         last_embeddings = self._get_last_embedding(sequence_embeddings, mask)
#         candidate_scores = last_embeddings @ self._item_embeddings.weight.T
#         candidate_scores[:, 0] = -torch.inf
#         candidate_scores[:, self._num_items + 1:] = -torch.inf
#         _, indices = torch.topk(candidate_scores, k=self._topk_k, dim=-1, largest=True)
#         return indices










