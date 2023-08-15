import argparse
import logging
import random
import os
from typing import List

import datasets
from tqdm import trange, tqdm

from dataset import Example
from plugin_dataset import PlugInDataset
from generate import Generator

logging.basicConfig(level=logging.INFO)

def pairwise_checking(all_generations: List[str], all_labels: List[str], all_examples: List[Example], label_map) -> List[str]:
    # Unplug list to hold the eids of examples to be unplugged
    unplug_list = []
    # Compare generations with labels
    for generation, label, example in zip(all_generations, all_labels, all_examples):
        # Check if generation is correct
        converted_label = label_map(label)
        if generation == converted_label:
            unplug_list.append(example.example_id)

    return unplug_list


def run_filtering(
        generator: Generator,
        args: argparse.Namespace
):
    # from converters.registry import get_converter
    random.seed(0)
    # dataset = datasets.load_dataset('sst2')
    # dataset = datasets.load_dataset('gsm8k', 'main')
    # dataset = datasets.load_dataset("iohadrubin/smcalflow")
    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name)
    # dataset["train"] = dataset["train"].select(range(1000))
    plugin_dataset = PlugInDataset(data_dict=dataset, data_type="train", src_key=args.src_key, tgt_key=args.tgt_key, batch_size=args.batch_size)
    # converter = get_converter(args.converter)

    from converters.math_word_problem_converter import MathConverter
    converter = MathConverter()

    # sufficient data selection
    cnt_removed = 0
    print(f"Initial Plugin Datset: {len(plugin_dataset)}")
    for i in trange(args.n_iter, desc="Iteration:"):
        plugin_dataset.shuffle()
        # Prepare all prompt inputs and labels
        all_labels = []
        all_examples = []
        all_generations = []
        for batch in tqdm(plugin_dataset.n_example_batch(n=2), total=len(plugin_dataset)//(args.batch_size*2)):
            if len(batch[-1]) < 2:
                break
            prompt_inputs = [converter.example2code(demos=group[:-1], target=group[-1]) for group in batch]
            # Batch Generation
            generations = generator.generate(
                prompt_inputs,
                decode_method=args.decode_method,
                # temperature=0.1,
                max_new_tokens=args.max_new_tokens,
                # num_batches_to_gen=args.num_batches_to_gen,
                num_generate=args.num_generate,
            )
            # print(generations)
            # for idx, group in enumerate(batch[:3]):
            #     print("*"*10)
            #     print("Target Label:", group[1].target_label)
            #     print("="*10)
            #     print("Input:", prompt_inputs[idx])
            #     print("="*10)
            #     print("Gen:", generations[idx])
            #     print("="*10)
            #     print("Pred:", converter.code2answer(generations[idx]))
            #     print("*"*10)
            all_labels.extend([group[-1].target_label for group in batch])
            all_examples.extend([group[-1] for group in batch])
            all_generations.extend([converter.code2answer(generation) for generation in generations])

        # Perform pairwise checking and unplug examples that don't pass the check
        unplug_ids = pairwise_checking(all_generations, all_labels, all_examples, converter.string2label)
        plugin_dataset.unplug(unplug_ids)
        print(f"{len(plugin_dataset.unplugged_data) - cnt_removed} removed in Round {i}.")
        cnt_removed = len(plugin_dataset.unplugged_data)
    print(f"Filtering finished. {len(plugin_dataset.all_data)} remained.")
    plugin_dataset.save_to_json(os.path.join(args.output_path, "plugin_set.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", required=True, help="dataset name", choices=["sst2", "gsm8k", "iohadrubin/smcalflow"]
    )
    parser.add_argument(
        "--dataset_config_name", required=True, help="dataset config name, e.g., main, smcalflow, etc."
    )
    parser.add_argument("--model_path", help="Path to the model", required=True)
    parser.add_argument(
        "--output_path", help="Path to the output directory (required for batch generation)"
    )

    parser.add_argument(
        "--max_length", help="max length to generate", type=int, required=False, default=None
    )
    parser.add_argument(
        "--batch_size", help="batch size to generate", type=int, required=False, default=1
    )
    parser.add_argument(
        "--n_iter", help="filter iterations", type=int, required=False, default=1
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
        "--converter", default="sentiment", help="converter"
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
    parser.add_argument("--add_io_sep", type=str, default="true", help="add io sep")
    parser.add_argument("--mode", type=str, default="plug", help="default or golden plugin")

    parser.add_argument(
        "--num_generate", type=int, default=1, help="number of generations to generate"
    )

    parser.add_argument("--interactive", action="store_true", help="interactive mode")

    parser.add_argument("--fp16", action="store_true", help="use fp16")

    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")

    parser.add_argument("--threshold", type=float, default=0.6, help="the threshold to select the golden set")

    parser.add_argument("--nocache", action="store_true", help="do not use cache")

    args = parser.parse_args()
    logging.info("model loading ...")

    args.add_io_sep = args.add_io_sep.lower() == "true"

    if args.max_length is not None and args.max_new_tokens is None:
        logging.warning(
            "max_new_tokens is not set, using max_length. We recommend using max_new_tokens to be compatible with huggingface"
        )
        args.max_new_tokens = args.max_length


    def train_func():
        generator = Generator(
            model_name=args.model_path,
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
        run_filtering(
            generator=generator,
            args=args,
        )
    
    from ray.train.torch import TorchTrainer
    from ray.air import ScalingConfig

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
    )

    trainer.fit()
