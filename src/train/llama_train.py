#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,set_seed
import json
from datasets import load_dataset, Features, Value
import datasets
from multiprocessing import cpu_count
import random
import wandb
import os
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb

random.seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    is_initialized: bool = field(default = False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_subset: str = field(default=None, metadata={"help": "train subset name if loading from huggingface datasets"})
    test_subset: str = field(default=None, metadata={"help": "test subset name if loading from huggingface datasets"})
    generator_or_evaluator: str = field(default="evaluator", metadata={"help": "whether to include query in the input for evaluating attribution"})
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "number of train samples."},
    )
    debug_setting: bool = field(default= False)
    contained_datasets: str = field(default='all', metadata={"help": "Contained datasets (e.g., ExpertQA, hagrid, etc. 'all' for containing all datasets.)"})
    dataset_version: str = field(default='v3.0', metadata={"help": "Contained datasets (e.g., ExpertQA, hagrid, etc. 'all' for containing all datasets.)"})
    template: str = field(default='base_c_e')
    template_path: str = field(default = 'src/train/template.json')
    def __post_init__(self):
        if self.generator_or_evaluator not in ["evaluator","generator"]:
            raise Exception("Should be either generator or evaluator")
        

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    def __init__(self, data_args: str, tokenizer: transformers.PreTrainedTokenizer, split='train'):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        # self.subset_name = data_args.train_subset if 'train' in split else data_args.test_subset

        self.num_train_samples = data_args.num_train_samples
        self.generator_or_evaluator = data_args.generator_or_evaluator
        self.input_ids, self.labels = self.load_and_tokenize_dataset(split, data_args)


    def _tokenize_fn(self, text: str, minus_len : int = 0) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length - minus_len,
                truncation=True,
            )

        input_ids = labels = tokenized.input_ids[0]
        input_ids_lens = labels_lens = tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def process_function(self, example):
        
        def format_prompt(example, have_question=False, have_response=False, prompt_name=self.data_args.template):
            query = example['question'] if example['question'] and example['question'] not in ["nan", "", None] else ""
            answer = example['claim'] if example['claim'] and example['claim'] not in ["nan", "", None] else ""
            response = example['response'] if example['response'] and example['response'] not in ["nan", "", None] else ""
            documents_concatenation = "\n\n\n".join(example["references"])

            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(query, answer, response, documents_concatenation)
            elif have_question and not have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(query, answer, documents_concatenation)
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "### Input:\nClaim: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, documents_concatenation)
            
            instructions = json.load(open(self.data_args.template_path))
            formatted_prompt = "{}{}".format(instructions[prompt_name]["llama2"], input)

            return formatted_prompt

        if self.generator_or_evaluator == "evaluator":
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

            source = format_prompt(example, have_question=have_question, have_response=have_response, prompt_name=self.data_args.template)
            target = f"{example['attribution_label']} {self.tokenizer.eos_token}"
            target_tokenized = self._tokenize_fn(target)
            len_target_tokenized = target_tokenized["input_ids_lens"] - 1
            source_tokenized = self._tokenize_fn(source, minus_len=len_target_tokenized)

            # source + target
            input_ids = torch.cat((source_tokenized["input_ids"], target_tokenized["input_ids"][-len_target_tokenized:]), dim=0)
            label = copy.deepcopy(input_ids)
            label[:-len_target_tokenized] = IGNORE_INDEX
        
            return {"input_ids": input_ids, "labels": label}
            

    def load_and_tokenize_dataset(self, split, data_args):
        features = Features({
            'question': Value('string'),  # 字符串字段
            'claim': Value('string'),  # 字符串字段
            'claim_raw_string': Value('string'),  # 字符串字段
            'response': Value('string'),  # 字符串字段
            'references': datasets.Sequence(Value("string")),  # 字符串字段
            'citation_links': datasets.Sequence(Value("string")),  # 字符串字段
            'webpage_references': datasets.Sequence(Value("string")),  # 字符串字段
            'attribution_label': Value('string'),  # 字符串字段
            'src_dataset': Value('string'),  # 字符串字段
            'id': Value('string'),  # 字符串字段
        })
        # Load the dataset
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = load_dataset(self.dataset_path, name=data_args.dataset_version, split="dev", features=features)
        else:
            dataset = load_dataset(self.dataset_path, name=data_args.dataset_version, split=split, features=features)
        # add data filter here (subset / delete some field / etc)
        if "train" in split:
            # if train set only contains 1 single dataset, then filter the others out from train split
            if data_args.contained_datasets in ['attributedqa_only', 'expertqa_only', 'stanford_only', 'hagrid_only']:
                if not isinstance(data_args.contained_datasets, list):
                    data_args.contained_datasets = [data_args.contained_datasets]
                # 使用filter函数过滤数据集
                dataset = dataset.filter(lambda example: any(dataset_name in example['src_dataset'].lower() for dataset_name in data_args.contained_datasets))
        # elif split == "stanford_dev":
        #     dataset = dataset.filter(lambda example : "stanford" in example['src_dataset'].lower())
        # elif split == "attributedqa_dev":
        #     dataset = dataset.filter(lambda example : "attributedqa" in example['src_dataset'].lower())
        # elif split == "hagrid_dev":
        #     dataset = dataset.filter(lambda example : "hagrid" in example['src_dataset'].lower())
        # elif split == "expertqa_dev":
        #     dataset = dataset.filter(lambda example : "expertqa" in example['src_dataset'].lower())
        
        # If num_train_samples is specified and less than the total dataset length
        if 0 < self.num_train_samples < len(dataset):
            dataset = dataset.select(range(self.num_train_samples))

        # Tokenize the dataset in a batched way
        tokenized_dataset = dataset.map(self.process_function, batched=False, num_proc=2)
        filtered_dataset = tokenized_dataset.filter(lambda example : any([ _ != -100 for _ in example["labels"]]), num_proc=2)
        logging.info(f"We cut {len(tokenized_dataset)} - {len(filtered_dataset)} instances")
        input_ids = [torch.tensor(d,dtype=torch.int64) for d in filtered_dataset['input_ids']]
        labels = [torch.tensor(l,dtype=torch.int64) for l in filtered_dataset['labels']]
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
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if not debug_setting:
        # if data_args.contained_datasets != "train":
        #     split_train = data_args.contained_datasets
        split_train = "train"
        split_eval = "dev"
        split_eval_ood = "test_ood"
    else:
        split_train = "train[:300]"
        split_eval = "dev[:30]"

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_train)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_eval)
    eval_dataset_stanford = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split="stanford_dev")
    eval_dataset_hagrid = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split="hagrid_dev")
    eval_dataset_attributedqa = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split="attributedqa_dev")
    eval_dataset_expertqa = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split="expertqa_dev")
    eval_dataset_ood = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_eval_ood)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_dataset_ood=eval_dataset_ood,
        eval_dataset_stanford=eval_dataset_stanford,
        eval_dataset_hagrid=eval_dataset_hagrid,
        eval_dataset_attributedqa=eval_dataset_attributedqa,
        eval_dataset_expertqa=eval_dataset_expertqa,
        data_collator=data_collator
    )

def train():
    # pdb.set_trace()
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # pdb.set_trace()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)
    print(training_args)
    # pdb.set_trace()
    with open(data_args.template_path) as f:
        template = json.load(f)
    global debug_setting
    global seed
    global prompter
    debug_setting = data_args.debug_setting
    if debug_setting:
        training_args.report_to = []
    seed = 42
    set_seed(seed)



    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )
    print('Start Loading Model')
    architectures = config.architectures[0]
    print(model_args.is_initialized)
    if not model_args.is_initialized:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        print("We initialize")
        model = transformers.AutoModelForCausalLM.from_config(
            config
        )
     
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None and "Causal" in architectures:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    global T_EOS_TOKEN
    T_EOS_TOKEN = tokenizer.eos_token_id

    print("before smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("after smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))

    def cal_acc(preds,labels):
        results = []
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                results.append(1)
            else:
                results.append(0)
        return round(100.0 * np.sum(results) / len(results), 4)
    
    def get_first_element(batch_index, seq_index):
        first_indexes = []
        for i in range(max(batch_index)):
            first_indexes.append(list(batch_index).index(i))

        return np.array(range(max(batch_index))),seq_index[np.array(first_indexes)]


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds = np.argmax(preds, axis= -1)
        answers_batch_idnex, answers_seq_idnex = np.where(np.logical_and(labels != -100, labels != T_EOS_TOKEN))
        answers_batch_idnex, answers_seq_idnex = get_first_element(answers_batch_idnex, answers_seq_idnex)
        compare_labels = labels[answers_batch_idnex,answers_seq_idnex]
        answers_seq_idnex = answers_seq_idnex - 1
        useful_preds = preds[answers_batch_idnex,answers_seq_idnex]
        acc = cal_acc(compare_labels, useful_preds)
        precision = precision_score(compare_labels, useful_preds, average='macro')
        recall = recall_score(compare_labels, useful_preds, average='macro')
        f1 = f1_score(compare_labels, useful_preds, average='macro')
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

    def preprocess_logits_for_metrics(logits, labels):
        if type(logits) == tuple:
            return logits[0].argmax(dim=-1)
        else:
            return logits.argmax(dim=-1)

    with training_args.main_process_first(desc="dataset map tokenization"):
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    class MultiEvalCallback(TrainerCallback):
        def __init__(self, trainer, eval_datasets, eval_steps, logging_steps):
            self.trainer = trainer
            self.eval_datasets = eval_datasets
            self.eval_steps = eval_steps
            self.logging_steps = logging_steps

        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # if state.global_step % self.logging_steps == 0 and hasattr(state, "training_loss"):
            #     wandb.log({"train_loss": state.training_loss})
            if state.global_step % self.eval_steps == 0:
                for dataset_name, eval_dataset in self.eval_datasets.items():
                    metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
                    for key, value in metrics.items():
                        log_name = f"{dataset_name}_{key}"
                        if training_args.local_rank <= 0:
                            wandb.log({log_name: value})  # 使用wandb直接记录指标
                        self.trainer.log_metrics(log_name, {log_name: value})
                        self.trainer.state.log_history.append({log_name: value})

    # training_args.report_to = []
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )
    # if training_args.local_rank <= 0:
    #     wandb.init()
    # callback = MultiEvalCallback(
    #     trainer=trainer,
    #     eval_datasets={
    #         "eval_stanford_set": data_module["eval_dataset_stanford"],
    #         "eval_expertqa_set": data_module["eval_dataset_expertqa"],
    #         "eval_attributedqa_set:": data_module["eval_dataset_attributedqa"],
    #         "eval_hagrid_set": data_module["eval_dataset_hagrid"],
    #         "eval_attrscore_set": data_module["eval_dataset_ood"],
    #         "eval_dev_set": data_module["eval_dataset"],
    #     },
    #     eval_steps=training_args.eval_steps,
    #     logging_steps=training_args.logging_steps,
    # )
    # trainer.add_callback(callback)
    
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(ignore_keys_for_eval = ["past_key_values"])
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    if torch.distributed.is_initialized():
        # 仅在主进程上保存模型
        if torch.distributed.get_rank() == 0:
            trainer.model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.model.save_pretrained(training_args.output_dir)
    else:
        trainer.model.config.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
