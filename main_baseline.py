import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, q_rubric_dict
import subprocess


def text_similarity(str1, str2):
    cmd = "perl Text-Similarity-0.13/bin/text_similarity.pl " + "--string " + "'" + str1 + "' '" + str2 + "'" + \
          " --type Text::Similarity::Overlaps" + " --normalize"
    out = subprocess.getoutput(cmd=cmd)
    return float(out) # Dice Coefficient


parser = argparse.ArgumentParser()

# Data args
parser.add_argument('--answer_data_path', type=str, default="./answers_train.csv",
                    help='Path to the input answer dataset')


args = parser.parse_args()

print("Loading Answer Data")
answers_datset = AnswersCSVDataset([args.answer_data_path])
print("Done")


for row in tqdm(answers_datset):
    a_id = row[0]  # answer id
    answer_line = row[1]  # answer text
    q_id = row[2]  # question id
    a_score = row[3]  # answer score
    a_feed = row[4]  # answer feedback

    # Get question text --> question_line
    question_reference = q_rubric_dict[q_id]

    dice_coefficients = [text_similarity(answer_line, a) for a in question_reference]

    avg_dice = sum(dice_coefficients) / len(dice_coefficients)

    # Do some inference on avg_dice
