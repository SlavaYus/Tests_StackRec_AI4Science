import torch.nn as nn


class LocalObjective:
    def __init__(self):
        # TODO: add scalable softmax
        self._loss = nn.CrossEntropyLoss()

    def __call__(self, inputs):
        all_logits = inputs["local_prediction"]  # (all_items, num_classes)
        all_labels = inputs["labels.ids"]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)

        return loss
