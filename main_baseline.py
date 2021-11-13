import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, q_rubric_dict
import subprocess
import re


def text_similarity(str1, str2):
    cmd = "perl Text-Similarity-0.13/bin/text_similarity.pl " + "--string " + "'" + str1 + "' '" + str2 + "'" + \
          " --type Text::Similarity::Overlaps" + " --normalize"

    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.stderr:
        print(cmd)
        print(out.stderr)
        exit(0)
    return float(out.stdout)  # Dice Coefficient


parser = argparse.ArgumentParser()

# Data args
parser.add_argument('--answer_data_path', type=str, default="./answers_train.csv",
                    help='Path to the input answer dataset')
parser.add_argument('--correct_threshold', type=float, default=.5)
parser.add_argument('--partial_threshold', type=float, default=.2)

args = parser.parse_args()

print("Loading Answer Data")
answers_datset = AnswersCSVDataset([args.answer_data_path])
print("Done")

total = 0
accurate = 0
for row in tqdm(answers_datset):
    a_id = row[0]  # answer id
    answer_line = row[1].strip()  # answer text
    q_id = row[2]  # question id
    a_score = row[3]  # answer score
    a_feed = row[4]  # answer feedback

    # Get question text --> question_line
    question_reference = q_rubric_dict[q_id]

    answer_line = re.sub(r'[^a-zA-Z0-9]', '', answer_line)

    dice_coefficients = [text_similarity(answer_line, re.sub(r'[^a-zA-Z0-9]', '', q))
                         for q in question_reference]

    avg_dice = sum(dice_coefficients) / len(dice_coefficients)

    correct_pred = avg_dice > args.correct_threshold
    partial_pred = avg_dice > args.partial_threshold
    accurate += (a_score == 'correct' and correct_pred) or \
                (a_score == 'partial correct' and partial_pred) or \
                (a_score == 'incorrect' and not correct_pred)

    total += 1

# Calculate Accuracy
print(f"Accuracy: {accurate/total}")

