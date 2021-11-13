import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, q_rubric_dict
import subprocess
import re


def text_similarity(str1, str2):
    """
    Wrapper method for TextSimilarity perl scripts
    :param str1: String to compare to str2
    :param str2: String to compare to str1
    :return: Dices coefficient of the two token sequences
    """
    cmd = "perl Text-Similarity-0.13/bin/text_similarity.pl " + "--string " + "'" + str1 + "' '" + str2 + "'" + \
          " --type Text::Similarity::Overlaps" + " --normalize"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return float(out.stdout)


# Arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--ata_path', type=str, default="./answers_train.csv",
                    help='Path to the input answer dataset')
parser.add_argument('--correct_threshold', type=float, default=.5)
parser.add_argument('--partial_threshold', type=float, default=.2)

args = parser.parse_args()

# Load Data
csv_dataset = AnswersCSVDataset([args.data_path])


total = 0
accurate = 0
for row in tqdm(csv_dataset):
    a_id = row[0]  # answer id
    answer_line = row[1].strip()  # answer text
    q_id = row[2]  # question id
    a_score = row[3]  # answer score
    a_feed = row[4]  # answer feedback

    # Get question text --> question_line
    question_reference = q_rubric_dict[q_id]

    # Clean answer text
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

