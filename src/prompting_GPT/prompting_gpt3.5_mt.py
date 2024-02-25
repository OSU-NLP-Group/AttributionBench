import re
from openai import OpenAI
import json
import numpy as np
import time
from datasets import load_dataset
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import random
import transformers
import json
from tqdm import tqdm
import traceback
import os
import concurrent.futures
import pdb

random.seed(42)

tokenizer = transformers.LlamaTokenizer.from_pretrained("Llama-2-7b-hf", use_fast=True, max_length=20000,)

with open("openai_api_key.txt", "r") as f:
    client = OpenAI(api_key=f.read().strip())

# add an argument parser and an argument named "test_subset_name"
parser = argparse.ArgumentParser()
parser.add_argument("--test_subset_name", type=str, default="test_all_subset_balanced")
parser.add_argument("--test_ood_subset_name", type=str, default="test_ood_all_subset_balanced")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106", help="gpt-3.5-turbo-1106, gpt-4-1106-preview")
parser.add_argument("--have_question", type=str, default="True")
parser.add_argument("--have_reference", type=str, default="False")
parser.add_argument("--prompt_name", type=str, default="base", help="base, info, breakdown, breakdown_strong, breakdown_weak")
parser.add_argument("--prompt_model_name", type=str, default="GPT", help="GPT")
args = parser.parse_args()

test_data = [json.loads(l) for l in open("../../data/{}.jsonl".format(args.test_subset_name))]
test_ood_data = [json.loads(l) for l in open("../../data/{}.jsonl".format(args.test_ood_subset_name) if args.test_ood_subset_name is not None else [])]
test_data += test_ood_data

global HAVE_QUESTION, HAVE_REFERENCE
HAVE_QUESTION = args.have_question == "True"
HAVE_REFERENCE = args.have_reference == "True"

if HAVE_QUESTION and HAVE_REFERENCE:
    prompt_name = f"{args.prompt_name}_q_c_e_r"
elif HAVE_QUESTION and not HAVE_REFERENCE:
    prompt_name = f"{args.prompt_name}_q_c_e"
elif not HAVE_QUESTION and HAVE_REFERENCE:
    prompt_name = f"{args.prompt_name}_c_e_r"
else:
    prompt_name = f"{args.prompt_name}_c_e"

output_file = "./gpt_generations/{}_{}_{}_{}.json".format(args.model, args.test_subset_name, prompt_name, args.prompt_model_name)


def format_prompt(example, have_question=False, have_response=False, prompt_name=prompt_name, tokenizer=tokenizer):
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
    
    instructions = json.load(open("prompts.json"))
    formatted_prompt = "{}{}".format(instructions[prompt_name][args.prompt_model_name], input)

    tokens = tokenizer.encode(formatted_prompt)
    if len(tokens) > 7000:
        tokens = tokens[:7000]
        formatted_prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    return formatted_prompt

# def get_attr_from_chatgpt(prompt, model="gpt-3.5-turbo"):
#     messages=[
#         {"role": "user", "content": prompt},
#     ]
#     retry_count = 0
#     while retry_count < 3:
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=messages,
#                 temperature=0,
#                 top_p=0.9,
#                 max_tokens=512,
#                 n=1
#             )
#             return response['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             print("Error:", str(e))
#             traceback.print_exc()
#             retry_count += 1
#             time.sleep(5)
#     print("Failed after 3 attempts")

res_key = '{}.eval.{}'.format(args.model, prompt_name)

def handle_exception(exc):
    print("Exception:", exc)

def get_attr_from_chatgpt_with_exception_handling(prompt, model="gpt-3.5-turbo"):
    try:
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=0.9,
            max_tokens=512,
            n=1
        )
        result = response.choices[0].message.content.strip()
        return result, None
    except Exception as e:
        return None, e

def process_example(example):
    prompt = format_prompt(example, have_question=HAVE_QUESTION, have_response=HAVE_REFERENCE)
    result, exception = get_attr_from_chatgpt_with_exception_handling(prompt, args.model)
    if exception is not None:
        for _ in range(5):
            time.sleep(1)
            result, exception = get_attr_from_chatgpt_with_exception_handling(prompt, args.model)
            print("result:", result, "exception:", exception)
            if exception is None:
                break
    example[res_key] = result if result is not None else ""


if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Skipping processing.")
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=36) as executor:
        for _ in tqdm(executor.map(process_example, test_data), total=len(test_data)):
            pass
json.dump(test_data, open(output_file,'a'))


def extract_pred_label(prediction):
    label_map = {"attributable": "attributable", "not attributable": "not attributable"}
    label_regex = r"|".join(list(label_map.keys()))

    pred_label = re.search(label_regex, prediction, re.IGNORECASE).group() if re.search(
            label_regex,
            prediction, re.IGNORECASE) is not None else 'None'

    pred_label = label_map[pred_label.lower()] if pred_label.lower() in label_map else "None"

    return pred_label

def evaluate_confusion_matrix(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precision[i] = true_positives / (true_positives + false_positives)
        recall[i] = true_positives / (true_positives + false_negatives)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    micro_true_positives = np.sum(np.diag(confusion_matrix))
    micro_false_positives = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)

    micro_f1 = micro_true_positives / (micro_true_positives + np.sum(micro_false_positives))
    macro_f1 = np.mean(f1)

    return precision, recall, f1, micro_f1, macro_f1


def get_metrics(output_file, src_dataset="in_domain"):
    res_key = '{}.eval.{}'.format(args.model, prompt_name)
    eval_examples = [example for example in json.load(open(output_file))]
    if src_dataset in ["ExpertQA", "Stanford-GenSearch", "AttributedQA", "LFQA", "BEGIN", "AttrScore-GenSearch", "HAGRID"]:
        eval_examples = [example for example in eval_examples if example['src_dataset'].lower() == src_dataset.lower()]
    elif src_dataset == "in_domain":
        eval_examples = [example for example in eval_examples if example['src_dataset'] in ["ExpertQA", "Stanford-GenSearch", "AttributedQA", "LFQA"]]
    elif src_dataset == "out_of_domain":
        eval_examples = [example for example in eval_examples if example['src_dataset'] in ["BEGIN", "AttrScore-GenSearch", "HAGRID"]]
    pred_labels = [extract_pred_label(example[res_key]) for example in eval_examples]
    true_labels = [example['attribution_label'] for example in eval_examples]
    acc = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=["attributable", "not attributable"])
    precision, recall, f1, micro_f1, macro_f1 = evaluate_confusion_matrix(conf_matrix)
    # pdb.set_trace()
    metric_str = ""
    metric_str += "Confusion matrix:\n{}\n".format(conf_matrix)
    metric_str += "Accuracy: {}\n".format(acc)
    metric_str += "Precision: {}\n".format(precision)
    metric_str += "Recall: {}\n".format(recall)
    metric_str += "F1: {}\n".format(f1)
    metric_str += "micro_f1: {}\n".format(micro_f1)
    metric_str += "macro_f1: {}\n".format(macro_f1)
    metric_str += "True attributable & pred not attributable: {}\n".format(conf_matrix[0, 1])
    metric_str += "True not attributable & pred attributable: {}\n".format(conf_matrix[1, 0])
    metric_str += "Total examples: {}\n".format(len(eval_examples))
    metric_str += "\n"
    return metric_str

with open("./eval_results/eval_results_{}_{}_{}_{}.txt".format(args.model, args.test_subset_name, prompt_name, args.prompt_model_name), "w") as f:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write("Current time: {}\n".format(current_time))
    f.write("Model: {}\n".format(args.model))
    f.write("Prompt: {}\n".format(prompt_name))
    f.write("Test set: {}\n".format(args.test_subset_name))
    f.write("Have question: {}\n".format(HAVE_QUESTION))
    f.write("Have reference: {}\n".format(HAVE_REFERENCE))
    f.write("Example: {}\n".format(format_prompt(test_data[0], have_question=HAVE_QUESTION, have_response=HAVE_REFERENCE)))
    f.write("Output file: {}\n".format(output_file))
    f.write("Metrics:\n\n")
    f.write("########## ExpertQA ##########\n")
    f.write(get_metrics(output_file, src_dataset="ExpertQA"))
    f.write("########## Stanford-GenSearch ##########\n")
    f.write(get_metrics(output_file, src_dataset="Stanford-GenSearch"))
    f.write("########## AttributedQA ##########\n")
    f.write(get_metrics(output_file, src_dataset="AttributedQA"))
    f.write("########## LFQA ##########\n")
    f.write(get_metrics(output_file, src_dataset="LFQA"))
    f.write("########## in_domain ##########\n")
    f.write(get_metrics(output_file, src_dataset="in_domain"))
    f.write("########## BEGIN ##########\n")
    f.write(get_metrics(output_file, src_dataset="BEGIN"))
    f.write("########## AttrScore-GenSearch ##########\n")
    f.write(get_metrics(output_file, src_dataset="AttrScore-GenSearch"))
    f.write("########## HAGRID ##########\n")
    f.write(get_metrics(output_file, src_dataset="HAGRID"))
    f.write("########## out_of_domain ##########\n")
    f.write(get_metrics(output_file, src_dataset="out_of_domain"))
    