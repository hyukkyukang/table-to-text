import os
import sys
import json
import tqdm
import argparse
import subprocess

from hkkang_utils import file as file_utils
from hkkang_utils import string as string_utils

from src.infer import Inferer
from src.data.totto_data import TottoDataset

sys.path.append('/home/table-to-text/evaluation/language_repo/')


def parse_eval_stdout(eval_str) -> str:
    bleu_scores = []
    parent_scores = []
    bleu_prefix = '"score": '
    parent_prefix = "Precision = "
    for line in eval_str.split("\n"):
        line = line.strip()
        if line.startswith(bleu_prefix):
            bleu_score = line[len(bleu_prefix):].strip(",")
            assert string_utils.is_number(bleu_score), f"score should be number, but found: {bleu_score}"
            bleu_scores.append(float(bleu_score))
        elif line.startswith(parent_prefix):
            line = line.strip()
            words = line.split(" ")
            precision = words[2]
            recall = words[5]
            fscore = words[8]
            assert string_utils.is_number(precision), f"precision should be number, but found: {precision}"
            assert string_utils.is_number(recall), f"recall should be number, but found: {recall}"
            assert string_utils.is_number(fscore), f"fscore should be number, but found: {fscore}"
            parent_scores.append({"precision": float(precision),
                                "recall": float(recall),
                                "fscore": float(fscore)})
    return bleu_scores, parent_scores


def evaluate_totto(prediction_path, target_path):
    eval_stdout = subprocess.run(["/bin/bash",
                             "evaluation/language_repo/language/totto/totto_eval.sh",
                             "--prediction_path",
                             prediction_path,
                             "--target_path",
                             target_path
                             ], capture_output=True).stdout.decode('utf-8')
    result = parse_eval_stdout(eval_stdout)
    return result


def main(args):
    # Path setting
    inference_out_path = os.path.join(args.output_dir_path, "inference_output.txt")
    inference_data_path = os.path.join(args.output_dir_path, "inference_data.txt")
    
    # Load inferer
    inferer = Inferer(args.model_path)

    # Load test data
    dataloader = TottoDataset.get_dataloader(inferer.model.tokenizer, *file_utils.split_path_into_dir_and_file_name(args.data_path))
    
    # Run inference
    print("Running inference...")
    inferred_texts =[]
    for batch in tqdm.tqdm(dataloader):
        assert len(batch.data) == 1, "Batch size should be 1"
        inferred_texts.append(inferer.inference(batch.data[0].input_str))
        break
    print("Inference done.")
    
    # Create directory to save output
    file_utils.create_directory(args.output_dir_path)
    
    # Write into a file
    inferred_texts = [inferred_texts[0] for _ in range(len(dataloader))]
    print("Writing eval data into a file...")
    with open(inference_data_path, "w") as f:
        for batch in dataloader:
            for datum in batch:
                f.write(json.dumps(datum.raw_datum)+"\n")    
    print("Writing result into a file...")
    with open(inference_out_path, "w") as f:
        for text in inferred_texts:
            f.write(text + "\n")
    print("Writing done.")

    
    # Evaluate
    print("Evaluating...")
    result = evaluate_totto(inference_out_path, inference_data_path)
    print("Evaluation done.")
    print(result)


def parse_argument():
    parser = argparse.ArgumentParser(description="Table-to-text inference")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved training context")
    parser.add_argument("--output_dir_path", type=str, default="tmp", help="Path to output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    main(args)
