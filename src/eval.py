import sys
import subprocess

from hkkang_utils import string as string_utils

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
