import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut.util import JSONParseEvaluator, load_json, save_json
from donut.fast_model import DonutModel

device = 'cuda'

def test(arg):
    evaluator = JSONParseEvaluator()
    dataset = load_dataset('naver-clova-ix/cord-v2', split='test')
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    pretrained_model.eval()
    if device =='cuda':
        pretrained_model = pretrained_model.half()
        pretrained_model = pretrained_model.cuda()

    predictions = []
    ground_truths = []
    accs = []

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])
        image=sample["image"]
        output = pretrained_model.inference(image=image, prompt=f"<s_cord-v2>")["predictions"][0]

        gt = ground_truth["gt_parse"]
        score = evaluator.cal_acc(output, gt)

        accs.append(score)
        predictions.append(output)
        ground_truths.append(gt)

    scores = {
            "ted_accuracies": accs,
            "ted_accuracy": np.mean(accs),
            "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
        }

    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    args = parser.parse_args()
    predictions = test(args)
