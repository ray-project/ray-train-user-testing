import random
from dataclasses import dataclass
import os
from dataset import Example
from json import JSONEncoder

import numpy as np
import json
from typing import List, Optional, Iterator, Dict, Tuple


class PlugInDataset:
    def __init__(
            self,
            data_dir=None,
            data_dict=None,
            data_type="train",
            src_key="",
            tgt_key="",
            batch_size=1,
            embedding_path=None
    ):
        if data_dir is not None:
            self.all_data = self._load_questions(os.path.join(
                data_dir, f'{data_type}.jsonl'), data_type=data_type, src_key=src_key, tgt_key=tgt_key)
        elif data_dict is not None:
            # assert data_dict is not None, f"either {data_dir} or {data_dict} should be provided."
            self.all_data = self._load_from_data_dict(data_dict, data_type=data_type, src_key=src_key, tgt_key=tgt_key)
        else:
            self.all_data = {}
        self.data_type = data_type
        self.batch_size = batch_size
        self.unplugged_data = []  # to store unplugged data
        self._load_embeddings(embedding_path)

    def shuffle(self) -> None:
        """Shuffle the examples in-place."""
        example_list = list(self.all_data.values())
        random.shuffle(example_list)
        self.all_data = {example.example_id: example for example in example_list}

    def _load_embeddings(self, emb_path) -> None:
        if emb_path is None:
            return

        emb_path = os.path.join(emb_path, f"{self.data_type}_source.npy")
        if not os.path.exists(emb_path):
            return

        all_indices, examples = [], []
        for k, v in self.all_data.items():
            idx = k.split("-")[1]
            all_indices.append(idx)
            examples.append(v)
        all_source_emb = np.load(emb_path)
        source_emb = np.take(all_source_emb, all_indices, axis=0)
        # print("total_indices", len(all_indices), "source_emb", source_emb.shape)
        # assert len(examples) == source_emb.shape[0]
        for i, ex in enumerate(examples):
            ex.source_emb = source_emb[i]
            assert isinstance(ex.source_emb, np.ndarray), f"{type(ex.source_emb)}"

    @staticmethod
    def _load_from_data_dict(data_dict, data_type, src_key, tgt_key) -> Dict[str, Example]:
        res = {}
        for idx, dic in enumerate(data_dict[data_type]):
            eid = f"{data_type}-{idx}"
            res[eid] = Example(
                example_id=eid,
                source_input=dic[src_key],
                target_label=dic[tgt_key],
            )
        return res

    @staticmethod
    def _load_questions(path, data_type, src_key, tgt_key) -> Dict[str, Example]:
        res = {}
        with open(path, 'r') as fin:
            for idx, line in enumerate(fin):
                dic = json.loads(line)
                eid = f"{data_type}-{idx}"
                res[eid] = Example(
                    example_id=eid,
                    source_input=dic[src_key],
                    target_label=dic[tgt_key],
                )
        return res

    def examples(self) -> Iterator[Example]:
        for example in self.all_data:
            yield example

    def example_batch(self) -> Iterator[List[Example]]:
        """Yields batches of examples."""
        batch = []
        for example in self.all_data.values():
            batch.append(example)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:  # remaining examples in case total number of examples is not a multiple of batch_size
            yield batch

    def n_examples(self, n=2) -> Iterator[Tuple[Example]]:
        """Yield pairs (or groups of n) of examples in a non-overlapping way."""
        iterator = iter(self.all_data.values())
        while True:
            group = tuple(next(iterator) for _ in range(n))
            if len(group) == n:
                yield group
            else:
                break

    def n_example_batch(self, n=2) -> Iterator[List[List[Example]]]:
        """Yields batches of n-sized non-overlapping groups of examples."""
        batch, group = [], []
        for example in self.all_data.values():
            group.append(example)
            if len(group) == n:
                batch.append(group)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                group = []
        if group:
            batch.append(group)
        if batch:
            yield batch

    def unplug(self, eids: List[str]) -> None:
        for eid in eids:
            if eid in self.all_data:
                self.unplugged_data.append(eid)
                del self.all_data[eid]

    def save_to_json(self, file_path: str):
        directory_path = os.path.dirname(file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        data_to_save = {
            "all_data": [example.__dict__ for example in self.all_data.values()],
            "unplugged_data": list(self.unplugged_data)
        }
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    @classmethod
    def load_from_json(cls, file_path: str, batch_size: int, embedding_path=None) -> 'PlugInDataset':
        with open(file_path, 'r') as f:
            data = json.load(f)
        # assert embedding_path is not None
        instance = cls(batch_size=batch_size)  # adjust batch_size or other parameters as needed

        # We need to convert dictionaries back to Example objects
        all_data = {ex_dict['example_id']: Example(**ex_dict) for ex_dict in data['all_data']}
        instance.all_data = all_data
        instance.unplugged_data = data['unplugged_data']
        instance._load_embeddings(emb_path=embedding_path)
        return instance

    def __len__(self):
        return len(self.all_data)
