import os
from typing import Dict
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Iterator

def read_jsonl_field(in_fp, field):
    entries = []
    with open(in_fp) as infile:
        for line in infile:
            j = json.loads(line)
            entries.append(j[field])
    return entries


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]


def encode_line(
    tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
):
    extra_kw = (
        {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    )
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        src_key="question",
        tgt_key="answer",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.in_fp = os.path.join(data_dir, f"{type_path}.jsonl")
        # Both val and dev are names for validation files.
        if type_path == "val" and not os.path.exists(self.in_fp):
            self.in_fp = os.path.join(data_dir, f"dev.jsonl")
        self.src_arr = read_jsonl_field(in_fp=self.in_fp, field=src_key)
        self.tgt_arr = read_jsonl_field(in_fp=self.in_fp, field=tgt_key)
        self.src_lens = self.get_char_lens(self.src_arr)
        self.tgt_lens = self.get_char_lens(self.tgt_arr)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.in_fp}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # index = index + 1  # linecache starts at 1
        assert (
            len(self.src_arr) >= index
        ), f"data: src_arr index out of bound. \nsrc_arr = \n{self.src_arr}"
        source_line = self.prefix + self.src_arr[index].rstrip("\n")
        assert (
            len(self.tgt_arr) >= index
        ), f"data: tgt_arr index out of bound. \ntgt_arr = \n{self.tgt_arr}"
        tgt_line = self.tgt_arr[index].rstrip("\n")
        assert (
            source_line
        ), f"empty source line {source_line} for index {index} -- corresponding to target line {tgt_line}"
        assert (
            tgt_line
        ), f"empty tgt line {tgt_line} for index {index} -- corresponding to source line {source_line}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(arr):
        return [len(x) for x in arr]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch


@dataclass
class Example:
    example_id: str
    source_input: str
    target_label: str
    source_emb: Optional[np.ndarray] = None
    target_emb: Optional[np.ndarray] = None


class AutoregLMDataset(Dataset):
    IO_SEP = "|||"

    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        src_key="question",
        tgt_key="answer",
        max_source_length: int = 512,
        max_target_length: int = 128,
        prefix="",
    ):
        super().__init__()
        self.input_file_path = os.path.join(data_dir, f"{type_path}.jsonl")
        # Both val and dev are names for validation files.
        self.type_path = type_path
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.prefix = prefix
        self.pad_token_id = self.tokenizer.pad_token_id
        self.io_sep_token_id = self.tokenizer(self.IO_SEP)["input_ids"]

        assert (
            len(self.io_sep_token_id) == 1
        ), f"{self.IO_SEP} should be a single special token, maybe you forgot to add it to your tokenizer?"
        self.io_sep_token_id = torch.Tensor(self.io_sep_token_id)
        self.eos_token_id_tensor = torch.Tensor([self.tokenizer.eos_token_id])
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_input_length = max_source_length + max_target_length + 2

        if self.type_path == "val" and not os.path.exists(self.input_file_path):
            self.input_file_path = os.path.join(data_dir, f"dev.jsonl")
        print(f"Loading {self.input_file_path}...")
        data = pd.read_json(self.input_file_path, orient="records", lines=True)
        self.source_lines = data[src_key].apply(lambda x: x.strip()).tolist()
        self.target_lines = data[tgt_key].apply(lambda x: x.strip()).tolist()

    def __len__(self):
        return (
            len(self.source_lines)
            if self.type_path == "train"
            else len(self.source_lines)
        )

    def __getitem__(self, index):

        src_line = self.prefix + self.source_lines[index].rstrip("\n")

        tgt_line = self.target_lines[index].rstrip("\n")
        # print("\nsrc_line:", src_line)
        # print("\n\n")
        # print("\ntgt_line:", tgt_line)
        # print("\n\n")
        source_ids = self.tokenizer(
            src_line,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        target_ids = self.tokenizer(
            tgt_line,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # print("\nsource_ids:", source_ids.shape)
        # print("\n\n")
        #
        # print("\ntarget_ids:", target_ids.shape)
        # print("\n\n")
        # input()

        x = torch.cat(
            [
                source_ids,
                self.io_sep_token_id,
                target_ids,
                self.eos_token_id_tensor,
            ],
            dim=0,
        )
        input_span = len(source_ids) + 1

        # the labels are everything after the input span. This is not standard language modeling, it's a seq2seq setup.
        # A similar strategy was used to train COMET with GPT-2.
        y = torch.cat([torch.Tensor([-100] * input_span), x[input_span:]], dim=0)
        # print(len(source_ids), x.shape, torch.Tensor([-100] * len(source_ids)).shape, x[len(source_ids):].shape, y.shape)

        assert x.shape == y.shape, f"{x.shape} != {y.shape}"

        if self.type_path == "train":
            return x.long(), y.long()
        else:
            return x.long(), y.long(), src_line, tgt_line

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        num_elems = len(batch)
        max_x_len = max([len(tup[0]) for tup in batch])
        max_x_len = min(max_x_len, self.max_input_length)
        batch_x = torch.zeros(num_elems, max_x_len).long()
        batch_y = torch.zeros(num_elems, max_x_len).long()
        batch_attention_mask = torch.zeros(num_elems, max_x_len).long()
        # token_type_ids = torch.zeros(num_elems, max_token_len).long()
        for i in range(num_elems):
            x = batch[i][0][:max_x_len]
            y = batch[i][1][:max_x_len]
            length = len(x)
            batch_x[i, :length] = torch.LongTensor(x)
            batch_y[i, :length] = torch.LongTensor(y)
            batch_attention_mask[i, :length] = 1
            batch_y[i, batch_attention_mask[i] == 0] = -100
        # print({"input_ids": batch_x, "attention_mask": batch_attention_mask, "labels": batch_y})
        return {
            "input_ids": batch_x,
            "attention_mask": batch_attention_mask,
            "labels": batch_y,
        }

    def collate_fn_eval(self, batch) -> Dict[str, torch.Tensor]:
        res = self.collate_fn(batch)
        source_lines = [x[2] for x in batch]
        target_lines = [x[3] for x in batch]
        return {
            **res,
            "source_text": source_lines,
            "target_text": target_lines,
        }
