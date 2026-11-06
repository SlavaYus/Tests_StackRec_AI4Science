import pandas as pd
import torch
from torch.utils.data import DataLoader


class SasrecProcessor:
    def __init__(self, pad_id, mode):
        assert mode in ["train", "validation", "test"]
        self.pad_id = pad_id
        self.mode = mode

    def _pad_sequence(self, all_sample_events, all_sample_lengths):
        batch_size = all_sample_lengths.shape[0]
        max_sequence_length = all_sample_lengths.max().item()

        padded_sequence = torch.full(
            (batch_size, max_sequence_length),
            fill_value=0.0,
            dtype=torch.long,
        )

        mask = (
            torch.arange(end=max_sequence_length).tile(batch_size, 1)
            < all_sample_lengths[:, None]
        )  # (batch_size, max_seq_len)
        padded_sequence[mask] = all_sample_events

        return padded_sequence, mask

    def __call__(self, batch):
        processed_batch = {
            "user.ids": [],
            "item.ids": [],
            "item.length": [],
            "labels.ids": [],
        }

        for sample in batch:
            processed_batch["user.ids"].append(sample["user.ids"])
            context_item_sequence = sample["item.ids"][:-1]
            processed_batch["item.ids"].extend(context_item_sequence)
            processed_batch["item.length"].append(len(context_item_sequence))
            if self.mode == "train":
                labels_item_sequence = sample["item.ids"][1:]
                processed_batch["labels.ids"].extend(labels_item_sequence)
            elif self.mode in ["validation", "test"]:
                label = sample["item.ids"][-1]
                processed_batch["labels.ids"].append(label)

        for key, value in processed_batch.items():
            processed_batch[key] = torch.tensor(value, dtype=torch.long)

        padded_sequence_ids, mask = self._pad_sequence(
            processed_batch["item.ids"], processed_batch["item.length"]
        )

        processed_batch["padded_sequence_ids"] = padded_sequence_ids
        processed_batch["mask"] = mask

        return processed_batch



class SequenceDataset:
    def __init__(self, data_path, max_sequence_length, mode):
        self._mode = mode
        self._index = []
        self._fill_index(data_path, max_sequence_length)

    def _fill_index(self, data_path, max_sequence_length):
        df = pd.read_csv(data_path)
        user_items = df.groupby("user_id")["item_id"].apply(list).to_dict()
        for user_idx, item_ids in sorted(list(user_items.items()), key=lambda x: x[0]):
            item_sequnce = item_ids[-max_sequence_length:]
            self._index.append(
                {
                    "user.ids": user_idx,
                    "item.ids": item_sequnce,
                }
            )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        sample = self._index[index]
        return sample



def get_dataloaders(config, dataset_meta, train_path, val_path, test_path):
    splits = ["train", "validation", "test"]
    datasets = {}
    for data_path, split in zip([train_path, val_path, test_path], splits):
        datasets[split] = SequenceDataset(
            data_path,
            config["dataset"]["max_sequence_length"],
            mode=split,
        )

    dataloaders = {}
    for split in splits:
        collator = SasrecProcessor(pad_id=0, mode=split)
        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            **config["dataloader"][split],
            collate_fn=collator,
        )

    return dataloaders
