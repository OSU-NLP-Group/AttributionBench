---
configs:
- config_name: subset_balanced
  description: >-
    The subset balanced version of the dataset. Each sub-dataset contains the
    same number of attributable labels and not attributable labels.
  data_files:
  - split: train
    path: train_all_subset_balanced.jsonl
  - split: dev
    path: dev_all_subset_balanced.jsonl
  - split: test
    path: test_all_subset_balanced.jsonl
  - split: test_ood
    path: test_ood_all_subset_balanced.jsonl
- config_name: overall_balanced
  description: >-
    The overall balanced version of the dataset. The whole set contains the same
    number of attributable labels and not attributable labels, but each
    sub-dataset does not.
  data_files:
  - split: train
    path: train_overall_balanced.jsonl
  - split: dev
    path: dev_all_subset_balanced.jsonl
  - split: test
    path: test_all_subset_balanced.jsonl
  - split: test_ood
    path: test_ood_all_subset_balanced.jsonl
- config_name: not_balanced
  description: >-
    The not balanced version of the dataset. The label distribution is the same
    as full data which is not balanced, but the data scale is sampled as
    comparable with the two label balanced version.
  data_files:
  - split: train
    path: merged_train_sampled.jsonl
  - split: dev
    path: dev_all_subset_balanced.jsonl
  - split: test
    path: test_all_subset_balanced.jsonl
  - split: test_ood
    path: test_ood_all_subset_balanced.jsonl
- config_name: full_data
  description: Full training data. The label distribution is not balanced.
  data_files:
  - split: train
    path: merged_train.jsonl
  - split: dev
    path: dev_all_subset_balanced.jsonl
  - split: test
    path: test_all_subset_balanced.jsonl
  - split: test_ood
    path: test_ood_all_subset_balanced.jsonl
license: apache-2.0
task_categories:
- text-classification
language:
- en
pretty_name: AttributionBench
size_categories:
- 10K<n<100K
---

# AttributionBench
Code and datasets for the paper "AttributionBench: How Hard is Automatic Attribution Evaluation?".

## Dataset
We constructed this dataset from multiple existing data sources in a unified format, in order to create a unified and diverse testbed for evaluating advanced attribution evaluation systems. The dataset contains both in-domain training set and id-domain and out-of-domain test set.

## Usage
```python
import datasets

features = datasets.Features({
  'question': datasets.Value('string'),
  'claim': datasets.Value('string'),
  'claim_raw_string': datasets.Value('string'),
  'response': datasets.Value('string'),
  'references': datasets.Sequence(datasets.Value("string")),
  'citation_links': datasets.Sequence(datasets.Value("string")),
  'webpage_references': datasets.Sequence(datasets.Value("string")),
  'attribution_label': datasets.Value('string'),
  'src_dataset': datasets.Value('string'),
  'id': datasets.Value('string'),
  })

# in-domain train (subset-balanced)
# possible values for 'name' field: ["subset_balanced", "overall_balanced", "not_balanced", "full_data"]
dataset = datasets.load_dataset("osunlp/AttributionBench", name="subset_balanced", split="train", features=features)

# in-domain eval/test (subset-balanced)
# dataset = datasets.load_dataset("osunlp/AttributionBench", name="subset_balanced", split="test", features=features)
dataset = datasets.load_dataset("osunlp/AttributionBench", name="subset_balanced", split="test", features=features)

# out-of-domain test (subset-balanced)
dataset = datasets.load_dataset("osunlp/AttributionBench", name="subset_balanced", split="test_ood", features=features)
```

## Dataset Structure
### Data Instances
```json
{
  "question":"Is the number of horses living on Easter Island twice the number of people?",
  "claim":"According to James Grant-Peterkin in his book \u201cA Companion to Easter Island\u201d, there are almost 3,000 horses on Easter Island. However, locals often claim that they have more horses than people. The population of Easter Island is about 6,000 inhabitants. So it seems that the number of horses living on Easter Island is not twice the number of people.",
  "claim_raw_string":"According to James Grant-Peterkin in his book \u201cA Companion to Easter Island\u201d, there are almost 3,000 horses on Easter Island. However, locals often claim that they have more horses than people. The population of Easter Island is about 6,000 inhabitants. So it seems that the number of horses living on Easter Island is not twice the number of people.",
  "response":"According to James Grant-Peterkin in his book \u201cA Companion to Easter Island\u201d, there are almost 3,000 horses on Easter Island. However, locals often claim that they have more horses than people. The population of Easter Island is about 6,000 inhabitants. So it seems that the number of horses living on Easter Island is not twice the number of people.",
  "references":[
    "It is worth mentioning the huge population of horses (about 6,000) that already outnumber people and roam free on the island."
  ],
  "citation_links":[],
  "webpage_references":[],
  "attribution_label":"not attributable",
  "src_dataset":"AttrScore-GenSearch",
  "id":"AttrScore-GenSearch_7234d6e9-1f51-4203-9587-f539e34d34f4"
}
```
### Data Fields
- ```question```: ```str``` The question proposed by the user.
- ```claim```: ```str``` Part of the response to the question. Could be one single sentence or multiple sentences.
- ```claim_raw_string```: ```str``` The raw string of the claim from the original datasets before being processed.
- ```response```: ```str``` The response to the question generated by LMs or generative search engines.
- ```references```: ```List[str]``` A list of documents or paragraphs which could support the claim.
- ```citation_links```: ```Optional[List[str]]``` Reserved field for citation links.
- ```webpage_references```: ```Optional[List[str]]``` Reserved field for the webpage contents of the reference links.
- ```attribution_label```: ```str``` "attributable" or "not attributable".
- ```src_dataset```: ```str``` The source dataset of the data item.
- ```id```: ```str``` The unique id for the data item in AttributionBench.