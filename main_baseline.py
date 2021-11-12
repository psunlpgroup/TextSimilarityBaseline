import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, QuestionsCSVDataset
import subprocess


def TextSimilarity(str1, str2):
    cmd = "perl Text-Similarity-0.13/bin/text_similarity.pl " + "--string " + "'" + str1 + "' '" + str2 + "'" + \
          " --type Text::Similarity::Overlaps" + " --normalize" + " --verbose"
    out = subprocess.getoutput(cmd=cmd)
    results = {}

    for line in out.splitlines():
        if ':' in line:
            l = line.split(':')
            results[l[0].rstrip().lstrip()] = float(l[1].rstrip().lstrip())

    return results


parser = argparse.ArgumentParser()

# Data args
parser.add_argument('--answer_data_path', type=str,
                    help='Path to the input answer dataset')

parser.add_argument('--question_data_path', type=str,
                    help='Path to input question dataset')

args = parser.parse_args()

# Get answer data from csv
print("Loading Answer Data")
# questions_dataset = QuestionsCSVDataset([args.answer_data_path])
print("Done")

# Get question data from csv
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
    question_line = " "

    r = TextSimilarity(answer_line, question_line)

    # Do some inference on r --> using Dice, Cosine, Lesk, or other measures
