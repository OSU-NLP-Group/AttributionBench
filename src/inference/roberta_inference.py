from transformers import GenerationConfig,\
    AutoConfig,\
    AutoModelForSeq2SeqLM,\
    AutoModelForCausalLM,\
    AutoTokenizer,\
    AutoModelForSequenceClassification
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets, Features, Value, Sequence
from pathlib import Path
import os
import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)
from tqdm import tqdm
from functools import partial
import random
random.seed(42)
import torch
import jsonlines
import numpy as np
import hashlib
from argparse import ArgumentParser
import sys
sys.path.append(".")
# from ..longlora_train.llama_attn_replace import replace_llama_attn
import pdb


def remove_slash(t):
    return t.replace("/", "_")

def batch_samples(l, bs):
    for i in range(0, len(l), bs):
        yield l[i:i+bs]


def _parse_prediction(label, method="attrbench", relax=False, w_rationale=False):
    label = str(label).lower().strip()
    
    if not relax and not w_rationale:
        if method == "attrbench":
            if label == "attributable":
                return 1
            elif label == "not attributable":
                return 0
            else:
                return -1
        if method == "autoais":
            if label == "attributable":
                return 1
            elif label == "not attributable":
                return 0
            else:
                return -1
        if method == "attrscore":
            if label == "Attributable".lower():
                return 1
            elif label in ["Contradictory".lower(),"Extrapolatory".lower()]:
                return 0
            else:
                return -1

    elif not w_rationale and relax:
        if method == "attrbench":
            if "attributable" in label.lower() and not "not attributable" in label.lower():
                return 1
            elif "not attributable" in label.lower():
                return 0
            else:
                return -1
        if method == "autoais":
            if label == "1":
                return 1
            elif label == "0":
                return 0
            else:
                return -1
    
    elif w_rationale:
        if method == "attrbench":

            label_tag = "#### Label"
            input_tag = "### Input"
            try:
                assert label_tag.lower() in label,"rationale is in ###Label format"
            except:
                return -1
            search_part = label.split(label_tag.lower())[1]
            if input_tag.lower() in search_part:
                search_part = search_part.split(input_tag.lower())[0]
            search_part = search_part.lower().strip()
            
            if "attributable" in search_part.lower() and not "not attributable" in search_part.lower():
                return 1
            elif "not attributable" in search_part.lower():
                return 0
            else:
                return -1

    else:
        raise NotImplementedError()


def parse_prediction(l, method, relax=False):
    ret = []
    for label in l:
        ret.append(_parse_prediction(label, method, relax))
    return ret

        
#******************************************************************

#******************************************************************
def format_example_for_autoais(evidence, claim):
    # return "premise: {} hypothesis: {}".format(evidence, claim)
    return "hypothesis: {}\npremise: {}".format(claim, evidence)


#******************************************************************

#******************************************************************
def format_example_for_attrscore(claim, reference):
    return "Claim: {} \n Reference: {}".format(claim, reference)

#******************************************************************


def icl_combination(args,test_datas,icl_datas,icl_process_func, source_map = True, replace_msg = ""):
    num_icl = args.num_icl
    icl_use_rationale = args.icl_use_rationale

    if icl_use_rationale:
        parse_icl = partial(_parse_prediction,w_rationale = True)
    else:
        parse_icl = _parse_prediction
    # preextract all datas from icl datas classified by source
    if source_map:
        icl_datas_d = {}
        for key in list(set(icl_datas["src_dataset"])):
            icl_datas_d[key] = icl_datas.filter(lambda example: example["src_dataset"] == key,load_from_cache_file = False) 
            icl_datas_d[f"{key}_positive"] = icl_datas.filter(lambda example: example["src_dataset"] == key and parse_icl(example["attribution_label"],method="attrbench") == 1,load_from_cache_file = False) 
            icl_datas_d[f"{key}_negative"] = icl_datas.filter(lambda example: example["src_dataset"] == key and parse_icl(example["attribution_label"],method="attrbench") == 0,load_from_cache_file = False) 
            
    else:
        icl_datas_all = {}
        icl_datas_all["positive"] = icl_datas.filter(lambda example : parse_icl(example["attribution_label"],method="attrbench") == 1,load_from_cache_file = False) 
        icl_datas_all["negative"] = icl_datas.filter(lambda example : parse_icl(example["attribution_label"],method="attrbench") == 0,load_from_cache_file = False) 

    combined_data = []
    for i in range(0,len(test_datas)):
        source = test_datas[i]["src_dataset"]
        example = test_datas[i]["processed_text"]
        if num_icl == 1:
            # no restriction on label balance for icl
            if source_map:
                tmp_icl_datas = icl_datas_d[source].select(random.sample(range(len(icl_datas_d[source])), num_icl))
            else:
                tmp_icl_datas = icl_datas.select(random.sample(range(len(icl_datas)),num_icl))
        elif num_icl == 2:
            # restriction on label balance for icl and must be 1:1
            if source_map:
                tmp_icl_datas_pos = icl_datas_d[f"{source}_positive"].select(random.sample(range(len(icl_datas_d[f"{source}_positive"])), int(num_icl/2)))
                tmp_icl_datas_neg = icl_datas_d[f"{source}_negative"].select(random.sample(range(len(icl_datas_d[f"{source}_negative"])), int(num_icl/2)))
                tmp_icl_datas = concatenate_datasets([tmp_icl_datas_pos,tmp_icl_datas_neg])
            else:
                tmp_icl_datas_pos = icl_datas_all["positive"].select(random.sample(range(len(icl_datas_all["positive"])),int(num_icl/2)))
                tmp_icl_datas_neg = icl_datas_all["negative"].select(random.sample(range(len(icl_datas_all["negative"])),int(num_icl/2)))
                tmp_icl_datas = concatenate_datasets([tmp_icl_datas_pos,tmp_icl_datas_neg])
            
        else:
            raise NotImplementedError()

        _icl_text_l = []
        for index in range(len(tmp_icl_datas)):
            _icl_text = icl_process_func(tmp_icl_datas[index])
            _icl_text_l.append(_icl_text["processed_text"])
        
        random.shuffle(_icl_text_l)
        icl_text = "\n\n".join (_icl_text_l)
        text = icl_text + "\n\n" + example
        combined_data.append(replace_msg + text.replace(replace_msg,""))

    return {"processed_text":combined_data, "src_dataset":test_datas["src_dataset"]}


class HelloWord:
    def __init__(self, model_name, args):
        self.args =args        
        self.bs = args.bs
        self.config = AutoConfig.from_pretrained(model_name)
        architectures = self.config.architectures[0]
        self.is_llama = "llama" in architectures.lower()
        self.is_mixtral = "mixtral" in architectures.lower()
        self.is_roberta = "roberta" in architectures.lower()

        if "roberta" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map="auto",
            )
            self.model.eval()
        
        elif "mixtral" in model_name:
            self.only_decoer = True
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # attn_implementation="flash_attention_2",   #You can use flash attention on your local GPU with specific libraries
            )
            self.tokenizer.pad_token = "!" #Not EOS, will explain another time.
            # CUTOFF_LEN = 256  #Our dataset has shot text
            # LORA_R = 8
            # LORA_ALPHA = 2 * LORA_R
            # LORA_DROPOUT = 0.1
            # lora_config = LoraConfig(
            #     r=LORA_R,
            #     lora_alpha=LORA_ALPHA,
            #     target_modules=[ "w1", "w2", "w3"],  #Only Training the "expert" layers
            #     lora_dropout=LORA_DROPOUT,
            #     bias="none",
            #     task_type="CAUSAL_LM"
            # )
            # self.model = get_peft_model(self.model, lora_config)
     
        elif "Conditional" in architectures:
            self.only_decoer = False
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "Causal" in architectures:
            self.only_decoer = True
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **self.args.kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # if self.tokenizer.pad_token is None:
            #     self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token = "!"
        self.model.eval()
        gen_kwargs = {"pad_token_id":self.tokenizer.pad_token_id, "eos_token_id":self.tokenizer.eos_token_id, "bos_token_id":self.tokenizer.bos_token_id}
        # self.gen_config = GenerationConfig(temperature = self.args.temperature, do_sample = False, max_new_tokens = self.args.max_new_tokens, **gen_kwargs)
        self.gen_config = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=self.args.max_new_tokens, **gen_kwargs)
    
    @torch.no_grad()
    def _inference(self,examples):
        print("self.is_roberta:", self.is_roberta)
        if self.is_roberta:
            predictions = []
            pbar = tqdm(total=len(examples), desc="do inference")
            for batch in batch_samples(examples, self.args.bs):
                tokenized_batch = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model(**tokenized_batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                pbar.update(self.bs)
            # convert predictions into labels
            # predictions = [self.model.config.id2label[x] for x in predictions]
            return [int(item) if isinstance(item, np.int64) else item for item in predictions]
        else:
            outputs = []
            pbar = tqdm(total=len(examples), desc="do inference")
            for batch in batch_samples(examples, self.bs):
                try:
                    tokenized_batch = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=self.args.max_length, return_token_type_ids=False).to(self.model.device)
                    # pdb.set_trace()
                    output = self.model.generate(**tokenized_batch, num_beams=1, do_sample=False, max_new_tokens=self.args.max_new_tokens, generation_config=self.gen_config)
                    if self.only_decoer and (self.is_llama or self.is_mixtral):
                        output = output[:, tokenized_batch["input_ids"].shape[-1]:]
                    else:
                        output = output
                    output_text = self.tokenizer.batch_decode(output,skip_special_tokens= True)   

                except:
                    print("run one by one")
                    output_text = []
                    for single_batch in batch:
                        single_tokenized_batch = self.tokenizer(single_batch, truncation=True, padding=True, return_tensors="pt", max_length=self.args.max_length, return_token_type_ids=False).to(self.model.device)
                        _output = self.model.generate(**single_tokenized_batch,generation_config = self.gen_config)
                        if self.only_decoer and (self.is_llama or self.is_mixtral):
                            _output = _output[:,single_tokenized_batch["input_ids"].shape[-1]:]
                        else:
                            _output = _output
                        output_text.extend(self.tokenizer.batch_decode(_output,skip_special_tokens= True))

                outputs.extend(output_text)
                pbar.update(self.bs)
            logger.info("raw_outputs")
            logger.info(outputs)
            return outputs
    
    def inference(self,examples):
        outputs = self._inference(examples)
        return {"raw_outputs": outputs}
    




class AutoAIS(HelloWord):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)

    def process_data(self, example, index=-1, add_label=False):
        # reference https://github.com/chaitanyamalaviya/ExpertQA/blob/ba0ede57f34b4f70ef35b051ce829b0706d4f7c7/modeling/auto_attribution/autoais.py#L252C24-L252C24
        question = example['question'] if example['question'] and example['question'] not in ["nan", ""] else ""
        claim = example['claim'] if example['claim'] and example['claim'] not in ["nan", ""] else ""
        documents_concatenation = "\n\n\n".join(example["references"])
        # premise = question + " " + documents_concatenation
        premise = documents_concatenation
        hypo = claim
        return {"processed_text": format_example_for_autoais(premise, hypo), "src_dataset": example["src_dataset"]}
    



class AttrScore(HelloWord):
    def __init__(self,model_name, args):
        super().__init__(model_name, args)

    def process_data(self, example, index=-1, add_label=False):
        # reference: from attrscore github
        # Claim: Who is the current CEO of Twitter? The current CEO of Twitter is Elon Musk \n Reference: Elon Musk is the CEO of Twitter. Musk took over as CEO in October 2022 following a back-and-forth affair in which the billionaire proposed to purchase the social media company for $44 billion, tried to back out, and then ultimately went through with the acquisition. After becoming CEO, former CEO Parag Agrawal, CFO Ned Segal, and legal affairs and policy chief Vijaya Gadde were all dismissed from the company.
       if self.is_llama:
            prefix_llama = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nVerify whether a given reference can support the claim. Options: Attributable, Extrapolatory or Contradictory.\n\n### Input:\n{}\n\n### Response:"
            question = example['question'] if example['question'] and example['question'] not in ["nan",""] else ""
            claim = example['claim'] if example['claim'] and example['claim'] not in ["nan",""] else ""
            documents_concatenation = "\n\n\n".join(example["references"])
            template = "Claim: {}\n\nReference: {}\n"
            text = prefix_llama.format(template.format(question + " " + claim,documents_concatenation))
       else:
            prefix = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\n"
            question = example['question'] if example['question'] and example['question'] not in ["nan",""] else ""
            claim = example['claim'] if example['claim'] and example['claim'] not in ["nan",""] else ""
            documents_concatenation = "\n\n\n".join(example["references"])
            claim = question + " " + claim
            reference = documents_concatenation
            text = prefix + format_example_for_attrscore(claim,reference)
       return {"processed_text":text,"src_dataset":example["src_dataset"]}


class AttrBench(HelloWord):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        self.retrieval_d = {}
        if args.retrieval_path != "":
            with jsonlines.open(args.retrieval_path) as f:
                for line in f:
                    self.retrieval_d[line["id"]] = line

    def process_data(self, example, index=-1, add_label=False):
        def format_prompt(example, have_question=False, have_response=False, prompt_name=self.args.template):
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
            
            instructions = json.load(open(args.template_path))
            formatted_prompt = "[INST] {} [/INST] {}".format(instructions[prompt_name]["llama2"], input)

            return formatted_prompt
        
        if "q_c_e_r" in self.args.template:
                have_question = True
                have_response = True
        elif "q_c_e" in self.args.template:
            have_question = True
            have_response = False
        elif "c_e_r" in self.args.template:
            have_question = False
            have_response = True
        else:
            have_question = False
            have_response = False

        source = format_prompt(example, have_question=have_question, have_response=have_response, prompt_name=self.args.template)
        
        if add_label:
            target = f"{example['attribution_label']}"
            return {"processed_text": source + target, "src_dataset": example["src_dataset"]}
        
        return {"processed_text": source, "src_dataset": example["src_dataset"]}


def main(args):
    print(args)
    if args.debug:
        args.split = [args.split[0] + '[:1%]']
    if args.method == "autoais":
        # "google/t5_xxl_true_nli_mixture
        evaluator = AutoAIS(args.model_name, args)
    if args.method == "attrscore":
        # "osunlp/attrscore-flan-t5-xl"
        evaluator = AttrScore(args.model_name, args)
    if args.method == "attrbench":
        if args.debug:
            evaluator = AttrBench("gpt2", args)
        else:
            evaluator = AttrBench(args.model_name, args)

    pbar = tqdm(total=len(args.split), desc="for loop for datasets")
    # pdb.set_trace()
    for split in args.split:
        pbar.update(1)
        output_path = os.path.join(args.output_dir, args.method + f"{remove_slash(args.model_name)}" + f"_{split}" + ".json")
        print("output_path:", output_path)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        features = Features({
            'question': Value('string'),  # 字符串字段
            'claim': Value('string'),  # 字符串字段
            'claim_raw_string': Value('string'),  # 字符串字段
            'response': Value('string'),  # 字符串字段
            'references': Sequence(Value("string")),  # 字符串字段
            'citation_links': Sequence(Value("string")),  # 字符串字段
            'webpage_references': Sequence(Value("string")),  # 字符串字段
            'attribution_label': Value('string'),  # 字符串字段
            'src_dataset': Value('string'),  # 字符串字段
            'id': Value('string'),  # 字符串字段
        })
        datas = load_dataset(args.data_path, name=args.dataset_version, split=split, features=features)
        print("len(datas):", len(datas))
        
        if args.source_only != "":
            # only do inference on the data belonging to this source
            datas = datas.filter(lambda example: example["src_dataset"] == args.source_only, load_from_cache_file=False) 
            print("Only consider {}".format(args.source_only))
            print("source_only",len(datas))

        label_name = "attribution_label"
        raw_labels = datas[label_name]
        src_datasets = datas["src_dataset"]
        postprocess_labels = parse_prediction(raw_labels, args.method)
        
        print("before process_data map", len(datas))
        processed_datas = datas.map(evaluator.process_data, remove_columns=datas.column_names, load_from_cache_file=False, with_indices=True, num_proc=1)
        # pdb.set_trace()
        if args.source_only != "":
            processed_datas = processed_datas.filter(lambda example: example["src_dataset"] == args.source_only,load_from_cache_file = False) 
        print("after process_data map", len(processed_datas))

        input_datas = processed_datas["processed_text"]

        if args.num_icl != -1:
            assert args.icl_split != ""
            train_datas = load_dataset(args.data_path,split=args.icl_split)
            process_icl = partial(evaluator.process_data, add_label = True)
            if "ood" in split:
                input_datas = icl_combination(args, processed_datas, train_datas, process_icl, source_map= False)["processed_text"]
            else:
                input_datas = icl_combination(args, processed_datas, train_datas, process_icl, source_map= True)["processed_text"]
        
        print("input_datas:", len(input_datas))
        outputs = evaluator.inference(input_datas)

        logger.info("processed output")
        logger.info(outputs)
        
    
        with open(output_path,"w") as f:
            for raw_label, postprocess_label, raw_output, src_dataset, input_text in zip(raw_labels, postprocess_labels, outputs["raw_outputs"], src_datasets, input_datas):
                json.dump(
                    dict(
                        raw_label=raw_label,
                        postprocess_label=postprocess_label,
                        raw_output=raw_output,
                        src_dataset=src_dataset,
                        input_text=input_text
                    ),
                    f
                )
                f.write("\n")
            

if __name__ == "__main__":
    parser = ArgumentParser(description="Inference about acc")
    parser.add_argument("--method", choices=["autoais", "attrscore", "attrbench", "gpt4"])
    parser.add_argument("--model_name",default="",type=str)
    parser.add_argument("--data_path",default = "osunlp/AttributionBench")
    parser.add_argument("--dataset_version", default="v2.1")
    parser.add_argument("--output_dir",default="./inference_results")
    parser.add_argument("--bs",default=2,type=int)
    parser.add_argument("--split",default="test",nargs = "+")
    parser.add_argument("--max_length", default= 1024, type = int)
    parser.add_argument("--num_icl",default = -1, type = int)
    parser.add_argument("--relax",action ="store_true")
    parser.add_argument("--max_new_tokens",default = 6, type = int)
    parser.add_argument("--temperature",default = 0.01, type = float)
    parser.add_argument("--icl_split",default = "", help = "split as the pool for ICL")
    parser.add_argument("--icl_use_rationale",action = "store_true",default = False)
    parser.add_argument("--source_only",default="",type = str,help = "only run inference when source == source_only")
    parser.add_argument("--template", type = str, default = "base_llama")
    parser.add_argument("--template_path", type = str, default = "src/template.json")
    parser.add_argument("--debug",action ="store_true")
    retrieval_group = parser.add_argument_group("retrieval")
    retrieval_group.add_argument('--retrieval_k', type=int, help='number of retrieved part from documents', default = 0)
    retrieval_group.add_argument('--retrieval_path', type=str,default = "")
    longlora_group = parser.add_argument_group("longlora")
    longlora_group.add_argument("--flash_attn",action ="store_true")

    args = parser.parse_args()

    # if args.flash_attn:
    #     replace_llama_attn(use_flash_attn=True, use_full=True)

    # if "longlora" in args.template:
    #     replace_llama_attn(use_flash_attn=True, use_full=True)

    with open(args.template_path) as f:
        template = json.load(f)

    kwargs = {}
    if args.method == "attrbench":
        kwargs["torch_dtype"] = torch.bfloat16
    args.kwargs = kwargs
    main(args)