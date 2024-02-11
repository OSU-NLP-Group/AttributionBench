import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
from datasets import load_dataset, Features, Value
import datasets
import json

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-xl")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dev_data_path: str = field(default=None, metadata={"help": "Path to the dev data."})
    dataset_version: str = field(
        default="v3.0",
        metadata={
            "help": "Contained datasets (e.g., ExpertQA, hagrid, etc. 'all' for containing all datasets.)"
        },
    )
    template: str = field(default="base_c_e")
    template_path: str = field(default="src/train/template.json")


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    logits = (
        eval_preds.predictions[0]
        if isinstance(eval_preds.predictions, tuple)
        else eval_preds.predictions
    )
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = [int(p.startswith(l)) for p, l in zip(decoded_preds, decoded_labels)]
    return {"accuracy": sum(result) / len(result)}


def _tokenize_fn(s: str, tokenizer: transformers.PreTrainedTokenizer, is_target=False):
    if is_target:
        token_ids = tokenizer(
            text_target=s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        token_ids = torch.where(token_ids == tokenizer.pad_token_id, -100, token_ids)
    else:
        token_ids = tokenizer(
            s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]

    return token_ids


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_tokenized = [_tokenize_fn(s, tokenizer) for s in sources]
    targets_tokenized = [_tokenize_fn(t, tokenizer, is_target=True) for t in targets]

    return dict(input_ids=sources_tokenized, labels=targets_tokenized)


class SupervisedDataset(Dataset):
    def __init__(
        self, data_args: str, tokenizer: transformers.PreTrainedTokenizer, split="train"
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        # self.subset_name = data_args.train_subset if 'train' in split else data_args.test_subset

        self.input_ids, self.labels = self.load_and_tokenize_dataset(split, data_args)

    def _tokenize_fn(self, text: str, minus_len: int = 0) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length - minus_len,
            truncation=True,
        )

        input_ids = labels = tokenized.input_ids[0]
        input_ids_lens = labels_lens = (
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def process_function(self, example):
        def format_prompt(
            example,
            have_question=False,
            have_response=False,
            prompt_name=self.data_args.template,
        ):
            query = (
                example["question"]
                if example["question"] and example["question"] not in ["nan", "", None]
                else ""
            )
            answer = (
                example["claim"]
                if example["claim"] and example["claim"] not in ["nan", "", None]
                else ""
            )
            response = (
                example["response"]
                if example["response"] and example["response"] not in ["nan", "", None]
                else ""
            )
            documents_concatenation = "\n\n\n".join(example["references"])

            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(
                    query, answer, response, documents_concatenation
                )
            elif have_question and not have_response:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, " ".join(query, answer))
                # input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}\n\n### Output:"
                # input = input_template.format(query, answer, documents_concatenation)
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, answer)

            instructions = json.load(open(self.data_args.template_path))
            # formatted_prompt = "{}{}".format(instructions[prompt_name]["llama2"], input)
            formatted_prompt = input

            return formatted_prompt

        if "q_c_e_r" in self.data_args.template:
            have_question = True
            have_response = True
        elif "q_c_e" in self.data_args.template:
            have_question = True
            have_response = False
        elif "c_e_r" in self.data_args.template:
            have_question = False
            have_response = True
        else:
            have_question = False
            have_response = False

        source = format_prompt(
            example,
            have_question=have_question,
            have_response=have_response,
            prompt_name=self.data_args.template,
        )
        target = "1" if f"{example['attribution_label']}" == "attributable" else "0"
        target_tokenized = self._tokenize_fn(target)
        source_tokenized = self._tokenize_fn(source)

        input_ids = source_tokenized["input_ids"]
        label = target_tokenized["input_ids"]
        # print("source:", source)
        # print("target:", target)
        # print("source tokens:", self.tokenizer.convert_ids_to_tokens(source_tokenized["input_ids"]))
        # print("target tokens:", self.tokenizer.convert_ids_to_tokens(target_tokenized["input_ids"]))
        return {"input_ids": input_ids, "labels": label}

    def load_and_tokenize_dataset(self, split, data_args):
        features = Features(
            {
                "question": Value("string"),  # 字符串字段
                "claim": Value("string"),  # 字符串字段
                "claim_raw_string": Value("string"),  # 字符串字段
                "response": Value("string"),  # 字符串字段
                "references": datasets.Sequence(Value("string")),  # 字符串字段
                "citation_links": datasets.Sequence(Value("string")),  # 字符串字段
                "webpage_references": datasets.Sequence(Value("string")),  # 字符串字段
                "attribution_label": Value("string"),  # 字符串字段
                "src_dataset": Value("string"),  # 字符串字段
                "id": Value("string"),  # 字符串字段
            }
        )
        # Load the dataset
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = load_dataset(
                self.dataset_path,
                name=data_args.dataset_version,
                split="dev",
                features=features,
            )
        else:
            dataset = load_dataset(
                self.dataset_path,
                name=data_args.dataset_version,
                split=split,
                features=features,
            )

        # Tokenize the dataset in a batched way
        tokenized_dataset = dataset.map(
            self.process_function, batched=False, num_proc=2
        )
        filtered_dataset = tokenized_dataset.filter(
            lambda example: any([_ != -100 for _ in example["labels"]]), num_proc=2
        )
        logging.info(
            f"We cut {len(tokenized_dataset)} - {len(filtered_dataset)} instances"
        )
        input_ids = [
            torch.tensor(d, dtype=torch.int64) for d in filtered_dataset["input_ids"]
        ]
        labels = [
            torch.tensor(l, dtype=torch.int64) for l in filtered_dataset["labels"]
        ]
        logging.info(f"{self.tokenizer.decode(input_ids[0],skip_special_tokens=True)}")
        return input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    split_train = "train"
    split_eval = "dev"
    split_eval_ood = "test_ood"

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_train
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_eval
    )
    eval_dataset_stanford = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="stanford_dev"
    )
    eval_dataset_hagrid = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="hagrid_dev"
    )
    eval_dataset_attributedqa = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="attributedqa_dev"
    )
    eval_dataset_expertqa = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="expertqa_dev"
    )
    eval_dataset_ood = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_eval_ood
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Suppress wandb
    training_args.report_to = []

    with open(data_args.template_path) as f:
        template = json.load(f)

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
