import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, q_rubric_dict
from sklearn.metrics import classification_report
from sklearn import tree
import subprocess
import re


def text_similarity(str1, str2):
    """
    Wrapper method for TextSimilarity perl scripts
    :param str1: String to compare to str2
    :param str2: String to compare to str1
    :return: Dices coefficient of the two token sequences
    """
    str1 = re.sub(r'[^a-zA-Z0-9 ]+', '', str1)
    str2 = re.sub(r'[^a-zA-Z0-9 ]+', '', str2)

    cmd = "perl Text-Similarity-0.13/bin/text_similarity.pl " + "--string " + "'" + str1 + "' '" + str2 + "'" + \
          " --type Text::Similarity::Overlaps" + " --verbose True"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    reg = r' *(Raw score|Precision|Recall|F\-measure|Dice|E\-measure|Cosine|Raw Lesk| Lesk)' \
          r' *: *[+-]?([0-9]*[.])?[0-9]+ *'

    similarity_features = {}
    for line in out.stderr.split("\n"):
        if re.match(reg, line):
            line = line.strip().split(':')
            similarity_features[line[0].strip()] = float(line[1].strip())

    return similarity_features


# Arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, default="./answers_train.csv",
                    help='Path to the train dataset')
parser.add_argument('--test_data_path', type=str, default="./answers_train.csv",
                    help='Path to the test dataset')

parser.add_argument('--method', type=str, default='threshold')
parser.add_argument('--correct_threshold', type=float, default=.2)
parser.add_argument('--partial_threshold', type=float, default=.1)

args = parser.parse_args()

cl_dict = {'correct': 0, 'partial correct': 1, 'incorrect': 2}

# Load Data
test_data = AnswersCSVDataset(args.test_data_path)

"""
    Fit on train set
"""
if args.method == 'tree':
    X = []
    Y = []


y_true = []
y_pred = []
debug_counter = 0
for row in tqdm(test_data):
    a_id = row[0]  # answer id
    answer_line = row[1].strip()  # answer text
    q_id = row[2]  # question id
    a_score = row[3]  # answer score
    a_feed = row[4]  # answer feedback

    # Get question text --> question_line
    question_reference = q_rubric_dict[q_id]

    # We are classifying based on answer score
    if a_score.isnumeric():
        a_score = int(a_score)
    else:
        a_score = cl_dict[a_score]
    y_true.append(a_score)


    """
        Classify answers using threshold on cosine similarity or decision tree
    """
    if args.method == 'threshold':
        avg = 0
        for q in question_reference:
            result = text_similarity(answer_line, q)
            if 'Cosine' in result:
                avg += result['Cosine']

        avg /= len(question_reference)

        if avg >= args.correct_threshold:
            y_pred.append(0)
        elif avg >= args.partial_threshold:
            y_pred.append(1)
        else:
            y_pred.append(2)

    elif args.method == 'tree':
        pass


# Calculate Accuracy
print(classification_report(y_true, y_pred))
