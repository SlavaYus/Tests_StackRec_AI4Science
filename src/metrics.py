# from collections import Counter

# import torch


# class BaseMetric:
#     def __init__(self, **kwargs):
#         pass


# class StatefullMetric(BaseMetric):
#     def reduce(self):
#         raise NotImplementedError


# class StaticMetric:
#     def __init__(self, name, value):
#         self._name = name
#         self._value = value

#     def __call__(self, inputs):
#         inputs[self._name] = self._value

#         return inputs


# class CompositeMetric:
#     def __init__(self, metrics):
#         self._metrics = metrics

#     def __call__(self, inputs):
#         for metric in self._metrics:
#             inputs = metric(inputs)
#         return inputs


# class NDCGMetric(BaseMetric):
#     def __init__(self, k, **kwargs):
#         super().__init__(**kwargs)
#         self._k = k

#     def __call__(self, inputs):
#         predictions = inputs["logits"][
#             :, : self._k
#         ].float()  # (batch_size, top_k_indices)
#         labels = inputs["labels.ids"].float()  # (batch_size)

#         assert labels.shape[0] == predictions.shape[0]

#         hits = torch.eq(
#             predictions, labels[..., None]
#         ).float()  # (batch_size, top_k_indices)
#         discount_factor = 1 / torch.log2(
#             torch.arange(1, self._k + 1, 1).float() + 1.0
#         ).to(
#             hits.device
#         )  # (k)
#         dcg = hits @ discount_factor  # (batch_size)

#         return dcg.cpu().tolist()


# class RecallMetric(BaseMetric):
#     def __init__(self, k, **kwargs):
#         super().__init__(**kwargs)
#         self._k = k

#     def __call__(self, inputs):
#         predictions = inputs["logits"][
#             :, : self._k
#         ].float()  # (batch_size, top_k_indices)
#         labels = inputs["labels.ids"].float()  # (batch_size)

#         assert labels.shape[0] == predictions.shape[0]

#         hits = torch.eq(
#             predictions, labels[..., None]
#         ).float()  # (batch_size, top_k_indices)
#         recall = hits.sum(dim=-1)  # (batch_size)

#         return recall.cpu().tolist()


# class CoverageMetric(StatefullMetric):
#     def __init__(self, k, num_items, **kwargs):
#         super().__init__(**kwargs)
#         self._k = k
#         self._num_items = num_items

#     def __call__(self, inputs):
#         predictions = inputs["logits"][:, : self._k]  # (batch_size, top_k_indices)
#         return predictions.reshape(-1).cpu().detach().tolist()  # (batch_size * k)

#     def reduce(self, values):
#         return len(set(values)) / self._num_items


# class NoveltyMetric(StatefullMetric):
#     """
#     Novelty(i) = 1 - (#users recommended i) / (#users that have NOT interacted with i)
#     """

#     def __init__(self, k, item2num_iteractions, num_users, **kwargs):
#         super().__init__(**kwargs)
#         self._k = k
#         self._item2num_iteractions = item2num_iteractions
#         self._num_users = num_users

#     def __call__(self, inputs):
#         predictions = inputs["logits"][:, : self._k]  # (batch_size, top_k_indices)
#         return predictions.reshape(-1).cpu().detach().tolist()  # (batch_size * k)

#     def reduce(self, recommended_items):
#         """
#         recommended_items: flat list of item ids for ALL users in epoch/batch
#         returns: average novelty over all items seen in recommendations
#         """

#         novelty_values = []
#         rec_count = Counter(recommended_items)

#         for item, recommended_times in rec_count.items():
#             # number of users who have NOT interacted with this item
#             no_interacted = self._num_users - self._item2num_iteractions[item]

#             if no_interacted == 0:
#                 continue  # skip items impossible to recommend

#             novelty = 1.0 - (recommended_times / no_interacted)
#             novelty_values.append(novelty)

#         if len(novelty_values) == 0:
#             return 0.0

#         return sum(novelty_values) / len(novelty_values)


# class ILDMetric(BaseMetric):
#     """
#     Vectorized ILD:
#     ILD(u) = mean over i<j of dist(item_i, item_j)
#     """

#     def __init__(self, k, dist_matrix, **kwargs):
#         super().__init__(**kwargs)
#         self._k = k
#         self._dist = dist_matrix  # (num_items, num_items)
#         self._triu = torch.triu_indices(k, k, offset=1)  # (2, K(K - 1)/2)

#     def __call__(self, inputs):
#         """
#         inputs["logits"]: (B, K) item ids
#         Returns: tensor (B,) — ILD per user
#         """
#         topk = inputs["logits"][:, : self._k]  # (B, K)

#         sub = self._dist[topk[:, None], topk[:, :, None]]  # (B, (K, K))

#         pairwise = sub[:, self._triu[0], self._triu[1]]  # (B, K(K - 1)/2)

#         ild_per_user = pairwise.mean(dim=1)  # (B,)

#         return ild_per_user.tolist()


from collections import Counter
import torch
import numpy as np


class BaseMetric:
    def __init__(self, **kwargs):
        pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class StaticMetric:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value
        return inputs


class CompositeMetric:
    def __init__(self, metrics):
        self._metrics = metrics

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


def compute_popularity_groups(item2num_interactions, num_items, n_groups=3):
    """
    Разделяет айтемы на группы так, чтобы в каждой группе было примерно 
    равное количество взаимодействий (не айтемов!).
    
    Args:
        item2num_interactions: dict {item_id: count}
        num_items: общее число айтемов
        n_groups: число групп (по умолчанию 3)
    
    Returns:
        dict {item_id: group_id} где group_id in {0, 1, ..., n_groups-1}
        0 = самые популярные, n_groups-1 = самые непопулярные
    """
    if not item2num_interactions:
        return {i: 0 for i in range(num_items + 2)}  # fallback
    
    # Сортируем айтемы по убыванию популярности
    sorted_items = sorted(
        item2num_interactions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Считаем общее число взаимодействий
    total_interactions = sum(count for _, count in sorted_items)
    target_per_group = total_interactions / n_groups
    
    # Распределяем айтемы по группам
    item2group = {}
    current_group = 0
    current_sum = 0
    
    for item_id, count in sorted_items:
        item2group[item_id] = current_group
        current_sum += count
        
        # Переходим к следующей группе, если набрали достаточно взаимодействий
        if current_group < n_groups - 1 and current_sum >= target_per_group * (current_group + 1):
            current_group += 1
    
    # Айтемы, которых нет в item2num_interactions (не было в train), 
    # относим к последней группе (непопулярные)
    for item_id in range(num_items + 2):
        if item_id not in item2group:
            item2group[item_id] = n_groups - 1
    
    return item2group


class NDCGMetric(BaseMetric):
    def __init__(self, k, num_items=None, item2num_interactions=None, n_groups=3, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._num_items = num_items
        self._item2group = None
        
        if item2num_interactions is not None and num_items is not None:
            self._item2group = compute_popularity_groups(
                item2num_interactions, num_items, n_groups
            )
        self._n_groups = n_groups

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k].float()
        labels = inputs["labels.ids"].float()
        
        assert labels.shape[0] == predictions.shape[0]
        
        hits = torch.eq(predictions, labels[..., None]).float()
        discount_factor = 1 / torch.log2(
            torch.arange(1, self._k + 1, 1).float() + 1.0
        ).to(hits.device)
        dcg = hits @ discount_factor  # (batch_size,)
        
        results = {"ndcg": dcg.cpu().tolist()}
        
        # Если есть группировка по популярности — считаем метрики по группам
        if self._item2group is not None:
            for group_id in range(self._n_groups):
                # Маска: какие примеры принадлежат этой группе
                mask = torch.tensor(
                    [self._item2group.get(int(lbl.item()), self._n_groups - 1) == group_id 
                     for lbl in labels],
                    dtype=torch.bool,
                    device=dcg.device
                )
                if mask.any():
                    group_dcg = dcg[mask].cpu().tolist()
                    results[f"ndcg_group{group_id}"] = group_dcg
                else:
                    results[f"ndcg_group{group_id}"] = []
        
        return results


class RecallMetric(BaseMetric):
    def __init__(self, k, num_items=None, item2num_interactions=None, n_groups=3, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._num_items = num_items
        self._item2group = None
        
        if item2num_interactions is not None and num_items is not None:
            self._item2group = compute_popularity_groups(
                item2num_interactions, num_items, n_groups
            )
        self._n_groups = n_groups

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k].float()
        labels = inputs["labels.ids"].float()
        
        assert labels.shape[0] == predictions.shape[0]
        
        hits = torch.eq(predictions, labels[..., None]).float()
        recall = hits.sum(dim=-1)  # (batch_size,)
        
        results = {"recall": recall.cpu().tolist()}
        
        # Если есть группировка по популярности — считаем метрики по группам
        if self._item2group is not None:
            for group_id in range(self._n_groups):
                mask = torch.tensor(
                    [self._item2group.get(int(lbl.item()), self._n_groups - 1) == group_id 
                     for lbl in labels],
                    dtype=torch.bool,
                    device=recall.device
                )
                if mask.any():
                    group_recall = recall[mask].cpu().tolist()
                    results[f"recall_group{group_id}"] = group_recall
                else:
                    results[f"recall_group{group_id}"] = []
        
        return results


class CoverageMetric(StatefullMetric):
    def __init__(self, k, num_items, num_users=None, item2num_interactions=None, n_groups=3, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._num_items = num_items
        self._num_users = num_users
        self._item2group = None
        
        if item2num_interactions is not None and num_items is not None:
            self._item2group = compute_popularity_groups(
                item2num_interactions, num_items, n_groups
            )
        self._n_groups = n_groups

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]  # (batch_size, k)
        labels = inputs["labels.ids"]  # (batch_size,)
        
        results = {}
        
        # Общая метрика — все рекомендованные айтемы
        all_items = predictions.reshape(-1).cpu().detach().tolist()
        results["coverage"] = all_items
        
        # По группам популярности (на основе ground truth labels)
        if self._item2group is not None:
            for group_id in range(self._n_groups):
                mask = [
                    self._item2group.get(int(lbl.item()), self._n_groups - 1) == group_id 
                    for lbl in labels
                ]
                # Берём рекомендации только для пользователей из этой группы
                group_items = []
                for i, m in enumerate(mask):
                    if m:
                        group_items.extend(predictions[i].cpu().detach().tolist())
                results[f"coverage_group{group_id}"] = group_items
        
        return results

    def reduce(self, values):
        # values — список рекомендованных айтемов
        if len(values) == 0:
            return 0.0
        return len(set(values)) / self._num_items


class NoveltyMetric(StatefullMetric):
    def __init__(self, k, item2num_iteractions, num_users, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._item2num_iteractions = item2num_iteractions
        self._num_users = num_users

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]
        return predictions.reshape(-1).cpu().detach().tolist()

    def reduce(self, recommended_items):
        novelty_values = []
        rec_count = Counter(recommended_items)

        for item, recommended_times in rec_count.items():
            no_interacted = self._num_users - self._item2num_iteractions.get(item, 0)
            if no_interacted == 0:
                continue
            novelty = 1.0 - (recommended_times / no_interacted)
            novelty_values.append(novelty)

        return sum(novelty_values) / len(novelty_values) if novelty_values else 0.0


class ILDMetric(BaseMetric):
    def __init__(self, k, dist_matrix, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._dist = dist_matrix
        self._triu = torch.triu_indices(k, k, offset=1)

    def __call__(self, inputs):
        topk = inputs["logits"][:, : self._k]
        sub = self._dist[topk[:, None], topk[:, :, None]]
        pairwise = sub[:, self._triu[0], self._triu[1]]
        ild_per_user = pairwise.mean(dim=1)
        return ild_per_user.tolist()



class MeanRankMetric(StatefullMetric):
    def __init__(self, k, num_items=None, item2num_interactions=None, n_groups=3, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._num_items = num_items
        self._item2group = None
        
        if item2num_interactions is not None and num_items is not None:
            self._item2group = compute_popularity_groups(
                item2num_interactions, num_items, n_groups
            )
        self._n_groups = n_groups

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]  # (batch_size, k)
        labels = inputs["labels.ids"]                 # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()
        has_hit = hits.sum(dim=1) > 0
        ranks = hits.argmax(dim=1).float() + 1.0
        ranks[~has_hit] = float(self._k + 1)

        results = {"mean_rank": ranks.cpu().tolist()}
        
        # Если есть группировка по популярности — считаем метрики по группам
        if self._item2group is not None:
            for group_id in range(self._n_groups):
                mask = torch.tensor(
                    [self._item2group.get(int(lbl.item()), self._n_groups - 1) == group_id 
                     for lbl in labels],
                    dtype=torch.bool,
                    device=ranks.device
                )
                if mask.any():
                    group_ranks = ranks[mask].cpu().tolist()
                    results[f"mean_rank_group{group_id}"] = group_ranks
                else:
                    results[f"mean_rank_group{group_id}"] = []
        
        return results

    def reduce(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)