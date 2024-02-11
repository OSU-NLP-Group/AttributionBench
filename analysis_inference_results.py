from argparse import ArgumentParser
import jsonlines
from collections import defaultdict as ddict
import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from functools import partial


def _parse_prediction(label, method="attrbench", relax=False, w_rationale=False):
    label = str(label).lower().strip()

    if not relax and not w_rationale:
        if method == "attrbench":
            if label.lower() == "attributable":
                return 1
            elif label.lower() == "not attributable":
                return 0
            else:
                return -1
        if method == "autoais":
            if label.lower() == "1":
                return 1
            elif label.lower() == "0":
                return 0
            else:
                return -1
        if method == "attrscore":
            if label.lower() == "Attributable".lower():
                return 1
            elif label.lower() in [
                "not attributable",
                "Contradictory".lower(),
                "Extrapolatory".lower(),
            ]:
                return 0
            else:
                return -1

    elif not w_rationale and relax:
        if method == "attrbench":
            if (
                "attributable" in label.lower()
                and not "not attributable" in label.lower()
            ):
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
                assert label_tag.lower() in label, "rationale is in ###Label format"
            except:
                return -1
            search_part = label.split(label_tag.lower())[1]
            if input_tag.lower() in search_part:
                search_part = search_part.split(input_tag.lower())[0]
            search_part = search_part.lower().strip()

            if (
                "attributable" in search_part.lower()
                and not "not attributable" in search_part.lower()
            ):
                return 1
            elif "not attributable" in search_part.lower():
                return 0
            else:
                return -1

    else:
        raise NotImplementedError()


def cal_acc(preds, labels):
    results = []
    for i in range(len(preds)):
        if preds[i] == labels[i] and preds[i] != -1:
            results.append(1)
        else:
            results.append(0)
    return round(np.sum(results) / len(results), 3)


def main(args):
    d = ddict(lambda: ddict(list))
    all_labels = []
    all_preds = []
    all_labels_neg = []
    all_preds_neg = []
    all_labels_pos = []
    all_preds_pos = []
    parse_func = partial(
        _parse_prediction,
        method=args.method,
        relax=args.relax,
        w_rationale=args.w_rationale,
    )
    with jsonlines.open(args.data_path) as f:
        for line in f:
            if line["src_dataset"].startswith("hagrid"):
                line["src_dataset"] = "hagrid"
            parsed_prediction = parse_func(line["raw_output"])

            # aovid some ill-defined problem
            if parsed_prediction == -1:
                parsed_prediction = 0 if line["postprocess_label"] == 1 else 1

            d[line["src_dataset"]]["labels"].append(line["postprocess_label"])
            d[line["src_dataset"]]["preds"].append(parsed_prediction)
            all_labels.append(line["postprocess_label"])
            all_preds.append(parsed_prediction)
            if line["postprocess_label"] == 0:
                all_labels_neg.append(line["postprocess_label"])
                all_preds_neg.append(parsed_prediction)
                d[line["src_dataset"]]["labels_neg"].append(line["postprocess_label"])
                d[line["src_dataset"]]["preds_neg"].append(parsed_prediction)
            if line["postprocess_label"] == 1:
                all_labels_pos.append(line["postprocess_label"])
                all_preds_pos.append(parsed_prediction)
                d[line["src_dataset"]]["labels_pos"].append(line["postprocess_label"])
                d[line["src_dataset"]]["preds_pos"].append(parsed_prediction)

    for key in d:
        d[key]["acc"] = cal_acc(d[key]["labels"], d[key]["preds"])
        d[key]["f1"] = f1_score(d[key]["labels"], d[key]["preds"], average="macro")
        d[key]["precision"] = precision_score(
            d[key]["labels"], d[key]["preds"], average="macro"
        )
        d[key]["recall"] = recall_score(
            d[key]["labels"], d[key]["preds"], average="macro"
        )

        y_true_filtered = [
            y for y, pred in zip(d[key]["labels"], d[key]["preds"]) if pred != -1
        ]
        y_pred_filtered = [pred for pred in d[key]["preds"] if pred != -1]

        d[key]["neg_acc"] = cal_acc(d[key]["labels_neg"], d[key]["preds_neg"])
        d[key]["neg_precision"] = precision_score(
            y_true_filtered, y_pred_filtered, pos_label=0
        )
        d[key]["neg_recall"] = recall_score(
            y_true_filtered, y_pred_filtered, pos_label=0
        )
        d[key]["neg_f1"] = f1_score(y_true_filtered, y_pred_filtered, pos_label=0)

        d[key]["pos_acc"] = cal_acc(d[key]["labels_pos"], d[key]["preds_pos"])
        d[key]["pos_precision"] = precision_score(
            y_true_filtered, y_pred_filtered, pos_label=1
        )
        d[key]["pos_recall"] = recall_score(
            y_true_filtered, y_pred_filtered, pos_label=1
        )
        d[key]["pos_f1"] = f1_score(y_true_filtered, y_pred_filtered, pos_label=1)

    d["all"]["acc"] = cal_acc(all_labels, all_preds)
    d["all"]["f1"] = f1_score(all_labels, all_preds, average="macro")
    d["all"]["precision"] = precision_score(all_labels, all_preds, average="macro")
    d["all"]["recall"] = recall_score(all_labels, all_preds, average="macro")

    y_true_all_filtered = [y for y, pred in zip(all_labels, all_preds) if pred != -1]
    y_pred_all_filtered = [pred for pred in all_preds if pred != -1]

    d["all"]["neg_acc"] = cal_acc(all_labels_neg, all_preds_neg)
    d["all"]["neg_precision"] = precision_score(
        y_true_all_filtered, y_pred_all_filtered, pos_label=0
    )
    d["all"]["neg_recall"] = recall_score(
        y_true_all_filtered, y_pred_all_filtered, pos_label=0
    )
    d["all"]["neg_f1"] = f1_score(y_true_all_filtered, y_pred_all_filtered, pos_label=0)

    d["all"]["pos_acc"] = cal_acc(all_labels_pos, all_preds_pos)
    d["all"]["pos_precision"] = precision_score(
        y_true_all_filtered, y_pred_all_filtered, pos_label=1
    )
    d["all"]["pos_recall"] = recall_score(
        y_true_all_filtered, y_pred_all_filtered, pos_label=1
    )
    d["all"]["pos_f1"] = f1_score(y_true_all_filtered, y_pred_all_filtered, pos_label=1)

    file_name, file_extension = os.path.splitext(args.data_path)
    data_path = f"{file_name}_analysis{file_extension}"
    with open(data_path, "w") as f:
        for key in d:
            json.dump(
                {
                    "src_dataset": key,
                    "f1": round(100 * d[key]["f1"], 1),
                    "acc": round(100 * d[key]["acc"], 1),
                    "precision": round(100 * d[key]["precision"], 1),
                    "recall": round(100 * d[key]["recall"], 1),
                    "neg_acc": round(100 * d[key]["neg_acc"], 1),
                    "neg_precision": round(100 * d[key]["neg_precision"], 1),
                    "neg_recall": round(100 * d[key]["neg_recall"], 1),
                    "neg_f1": round(100 * d[key]["neg_f1"], 1),
                    "pos_acc": round(100 * d[key]["pos_acc"], 1),
                    "pos_precision": round(100 * d[key]["pos_precision"], 1),
                    "pos_recall": round(100 * d[key]["pos_recall"], 1),
                    "pos_f1": round(100 * d[key]["pos_f1"], 1),
                },
                f,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser("analysis")
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--w_rationale", action="store_true")
    parser.add_argument(
        "--method", choices=["autoais", "attrscore", "attrbench", "gpt4"]
    )
    args = parser.parse_args()
    main(args)
