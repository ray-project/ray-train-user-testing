import os
import sys
from typing import Dict
import argparse

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
import logging
from tqdm import tqdm
import json

from dataset import trim_batch
from dataset import AutoregLMDataset

logging.basicConfig(level=logging.INFO)

import deepspeed

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Generator:
    def __init__(
        self,
        model_name,
        model_path,
        from_config: bool = False,
        config_name: str = None,
        is_autoreg: bool = True,
        batch_size: int = 32,
        fp16: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device = {self.device}, loading model from {model_path}")
        self.is_autoreg = is_autoreg
        self.config_name = config_name
        self.fp16 = fp16

        if from_config:
            self.load_from_config(config_name, model_path)
        else:
            self.load_pretrained(model_name, model_path)

        if self.is_autoreg:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.do_clean_output = True
            # self.io_sep_token = AutoregLMDataset.IO_SEP
            # self.tokenizer.add_tokens([AutoregLMDataset.IO_SEP])
            # self.io_sep_token_id = self.tokenizer(AutoregLMDataset.IO_SEP)["input_ids"][0]
            # print(f"Successfully loaded tokenizer")
            # print(f"Succesfully added {AutoregLMDataset.IO_SEP} to tokenizer.")
            
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"Warning: tokenizer vocab size ({len(self.tokenizer)}) does not match model vocab size ({self.model.config.vocab_size}).")
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            # check that the special tokens have been added to the tokenizer
            # num_toks = self.tokenizer([AutoregLMDataset.IO_SEP])["input_ids"][0]
            # assert (
            #     len(num_toks) == 1
            # ), f"IO_SEP token not added to tokenizer: {self.tokenizer([AutoregLMDataset.IO_SEP])}"

        ds_engine = deepspeed.init_inference(self.model, mp_size=4, dtype=torch.half, checkpoint=None)
        self.model = ds_engine.module

        self.batch_size = batch_size

        self.decoder_start_token_id = None

        if hasattr(self.model.config, "n_positions"):
            self.max_context_length = self.model.config.n_positions
        elif hasattr(self.model.config, "n_ctx"):
            self.max_context_length = self.model.config.n_ctx
        else:
            self.max_context_length = 4096

    def load_from_config(self, config_name: str, model_path: str):
        assert self.is_autoreg, "Loading from config only works for autoregressive models for now."
        config = AutoConfig.from_pretrained(config_name)
        model = AutoModelForCausalLM.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config_name)

        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        if self.fp16:
            model = model.half()
        # self.model = model.to(self.device).eval()
        self.model = model.eval()
        print(f"Successfully loaded model from {model_path}")

    def load_pretrained(self, model_name: str, model_path):
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError:
            # sometimes, only the model is saved, not the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config_name)
        if self.is_autoreg:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            if self.fp16:
                self.model = self.model.half()
            # self.model.to(self.device).eval()
            self.model.eval()

        else:
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device).eval()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
            self.do_clean_output = False

    def generate(
        self,
        input_txt,
        decode_method="beam",
        add_scores: bool = False,
        num_generate=5,
        temperature: float = 1.0,
        max_new_tokens: int = 150,
        num_batches_to_gen: int = None,
        num_return_sequences: int = None,
        outfile: str = None,
        cache_every: int = 5,
    ) -> Dict:
        # input_to_output_txt = dict()
        input_to_output_txt = []
        # input_txt = list(set(input_txt))  # we don't want to generate the same input twice

        # start a cache file for the generated outputs that persists across runs
        cache_file = None if outfile is None else outfile + ".cache"
        # print(f"Generating {num_generate} {decode_method} outputs for {len(input_txt)} inputs.")
        if cache_file is not None:
            print(f"Saving cache to {cache_file}.")

        # cache = None
        # if cache_file is not None:
        #
        #     if os.path.exists(cache_file):
        #         with open(cache_file, "r") as f:
        #             cache = json.load(f)
        #         input_to_output_txt.update(cache)
        #         input_txt = [i for i in input_txt if i not in input_to_output_txt]
        #         print(f"Skipping {len(cache)} inputs that have already been generated.")
        #     else:
        #         cache = dict()

        with torch.no_grad():

            method_to_kwargs = {
                "beam": {
                    "num_beams": num_generate,
                    "early_stopping": True,
                    "num_return_sequences": num_return_sequences
                    if num_return_sequences is not None
                    else num_generate,
                },
                "greedy-add-scores": {
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                },
                "greedy": {"do_sample": False},
                "sample": {
                    "do_sample": True,
                    "num_return_sequences": num_return_sequences
                    if num_return_sequences is not None else num_generate

                }
            }
            
            common_kwargs = {
                "temperature": temperature,
                # "top-p":1.0
            }
            kwargs = method_to_kwargs[decode_method]
            kwargs.update(common_kwargs)

            if add_scores and decode_method == "greedy":
                decode_method = "greedy-add-scores"

            scores = []
            batch_idx = 0

            # for input_txt_batched in tqdm(
            #     list(chunks(input_txt, self.batch_size)),
            #     desc=f"Performing {decode_method} generation",
            # ):
            for input_txt_batched in list(chunks(input_txt, self.batch_size)):
                if num_batches_to_gen is not None and batch_idx >= num_batches_to_gen:
                    break

                with torch.cuda.amp.autocast():
                    input_ids = self.tokenizer(
                        input_txt_batched,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                    ).to(self.device)
                    
                    input_ids, attention_mask = trim_batch(
                        **input_ids, pad_token_id=self.tokenizer.pad_token_id
                    )

                    if input_ids.shape[1] >= self.max_context_length - max_new_tokens:
                        # trim each input to the max context length
                        input_ids = input_ids[:, : self.max_context_length - max_new_tokens]
                        attention_mask = attention_mask[
                            :, : self.max_context_length - max_new_tokens
                        ]

                    

                    outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                # decoder_start_token_id=self.tokenizer.eos_token_id,
                                **kwargs,
                                )
                if add_scores:
                    outputs = outputs["sequences"]
                    batch_scores = outputs["sequences_scores"]

                decoded_ops = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                # if self.do_clean_output:
                #     decoded_ops = [self.clean_output(d) for d in decoded_ops]

                if (num_return_sequences and num_return_sequences > 1) or (num_generate and num_generate > 1):
                    num_return_sequences = (
                        num_return_sequences if num_return_sequences is not None else num_generate
                    )
                    grouped_generations = []
                    for i in range(0, len(decoded_ops), num_return_sequences):
                        grouped_generations.append(decoded_ops[i : i + num_return_sequences])
                    decoded_ops = grouped_generations

                for output_txt in decoded_ops:
                    # if cache_file is not None:
                    #     cache[input_txt] = output_txt
                    # input_to_output_txt[input_txt] = output_txt
                    input_to_output_txt.append(output_txt)
                # if batch_idx % cache_every == 0 and cache_file is not None:
                #     with open(cache_file, "w") as f:
                #         json.dump(cache, f)

                if add_scores:
                    scores.extend(batch_scores.detach().cpu().tolist())
                batch_idx += 1

            # if cache_file is not None:
            #     with open(cache_file, "w") as f:
            #         json.dump(cache, f)

            if add_scores:
                return input_to_output_txt, scores
            else:
                return input_to_output_txt

    def clean_output(self, text):
        # NOTE: If you are used to the OpenAI APIs, this might look strange.
        # By default, huggingface includes the prompt in the generations. Here we are
        # trimming it out.
        # return text
        if self.is_autoreg:
            text = text.split(self.io_sep_token)[-1]
        return text.strip()

    def interactive(self):
        input_txt = input("> ")
        max_new_tokens = int(input("max tokens>"))
        while input_txt != "exit":
            if self.is_autoreg:
                input_txt = input_txt #+ AutoregLMDataset.IO_SEP
            batch = self.tokenizer(
                [input_txt], return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)
            print(f"> Input: {input_txt}")
            # batch = input_ids, attention_mask = trim_batch(
            #     **batch, pad_token_id=self.tokenizer.pad_token_id
            # )
            input_ids = self.tokenizer.encode(input_txt, return_tensors='pt').cuda()

            print(f"> Input IDS: {input_ids}")
            outputs = self.model.generate(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                # do_sample=True,
                # temperature=0.9,
            )
            print(f"> Output Tokens: {outputs}")
            print(
                f"> Output: {self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)}"
            )
            print("=============================================")
            input_txt = input("> ")


def run_generation(
    generator: Generator,
    args: argparse.Namespace,
):

    if "jsonl" in args.source_path:
        data = pd.read_json(args.source_path, orient="records", lines=True)
        input_data = data[args.src_key].tolist()
    elif "txt" in args.source_path or "csv" in args.source_path:
        data = pd.read_csv(args.source_path, sep="\t", header=None)
        input_data = data[0].tolist()
    else:
        raise ValueError(f"Unknown source path: {args.source_path}")

    if generator.is_autoreg and args.add_io_sep:
        input_data = [x + AutoregLMDataset.IO_SEP for x in input_data]

    if args.num_batches_to_gen is not None:
        data = data.iloc[: args.num_batches_to_gen * generator.batch_size]

    generations = generator.generate(
        input_data,
        decode_method=args.decode_method,
        max_new_tokens=args.max_new_tokens,
        add_scores=args.add_scores,
        num_batches_to_gen=args.num_batches_to_gen,
        num_generate=args.num_generate,
        outfile=args.output_path,  # to create a cache for intermediate generations
    )
    inputs_to_generations = {}
    assert len(input_data) == len(generations)
    if generator.is_autoreg:
        for k, v in zip(input_data, generations):
            v = v.replace("<|endoftext|>", "")
            inputs_to_generations[k.split(AutoregLMDataset.IO_SEP)[0]] = v.split(AutoregLMDataset.IO_SEP)[1]
    else:
        for k, v in zip(input_data, generations):
            inputs_to_generations[k] = v

    # add a column to allow adding a list for beam
    data[f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"] = None
    data[f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"] = data[
        f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"
    ].astype("object")

    for i, row in data.iterrows():
        input_text = row[args.src_key]

        if input_text in inputs_to_generations:
            data.at[
                i, f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"
            ] = inputs_to_generations[input_text]
        else:
            data.at[i, f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"] = None

    exact_match = len(
        data[
            data[f"{args.decode_method}_generated_{args.tgt_key}_from_{args.src_key}"]
            == data[f"{args.tgt_key}"]
        ]
    ) / len(data)

    logging.info(f"Exact match = {(exact_match * 100):3f}%")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    data.to_json(os.path.join(args.output_path, "output.jsonl"), orient="records", lines=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model name", default="EleutherAI/gpt-neo-1.3B")

    parser.add_argument("--model_path", help="Path to the model", required=True)
    parser.add_argument(
        "--source_path", help="Path to the test jsonl file (required for batch generation)"
    )
    parser.add_argument(
        "--output_path", help="Path to the output file (required for batch generation)"
    )
    parser.add_argument(
        "--max_length", help="max length to generate", type=int, required=False, default=None
    )
    parser.add_argument(
        "--max_new_tokens",
        help="max new tokens to generate",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--src_key", default="question", help="source key in jsonl  (required for batch generation)"
    )
    parser.add_argument(
        "--tgt_key", default="answer", help="target key in jsonl  (required for batch generation)"
    )
    parser.add_argument("--add_scores", action="store_true", help="add scores to output")
    parser.add_argument(
        "--decode_method",
        default="greedy",
        help="decode method",
        choices=["greedy", "beam", "sample"],
    )
    parser.add_argument("--from_config", action="store_true", help="load from config")
    parser.add_argument("--config_name", type=str, help="Name of the config to use")
    parser.add_argument(
        "--is_autoreg",
        action="store_true",
        help="is the model autoregressive",
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_batches_to_gen",
        type=int,
        default=None,
        help="number of batches to generate",
    )
    
    parser.add_argument("--add_io_sep", type=str, default="true", help="add io sep")
    
    
    parser.add_argument(
        "--num_generate", type=int, default=1, help="number of generations to generate"
    )

    parser.add_argument("--interactive", action="store_true", help="interactive mode")

    parser.add_argument("--fp16", action="store_true", help="use fp16")
    
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    
    parser.add_argument("--nocache", action="store_true", help="do not use cache")

    args = parser.parse_args()
    logging.info("model loading ...")


    args.add_io_sep = args.add_io_sep.lower() == "true"
    
    if args.max_length is not None and args.max_new_tokens is None:
        logging.warning(
            "max_new_tokens is not set, using max_length. We recommend using max_new_tokens to be compatible with huggingface"
        )
        args.max_new_tokens = args.max_length

    generator = Generator(
        model_name=args.model_name,
        model_path=args.model_path,
        from_config=args.from_config,
        config_name=args.config_name,
        is_autoreg=args.is_autoreg,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    generator.model.zero_grad()
    generator.model.eval()
    logging.info("model loaded")

    logging.info(f"Model tokenizer length = {len(generator.tokenizer)}")
    if args.interactive:
        generator.interactive(max_new_tokens=args.max_new_tokens)
    else:
        run_generation(
            generator=generator,
            args=args,
        )

