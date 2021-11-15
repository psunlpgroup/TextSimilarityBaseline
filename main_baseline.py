import argparse
from tqdm import tqdm
from dataset import AnswersCSVDataset, q_rubric_dict
from sklearn.metrics import classification_report
from sklearn import tree
import subprocess
import re

cl_dict = {'correct': 0, 'partial correct': 1, 'incorrect': 2}


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


def get_text_features(answer, reference_answers, metrics=('Raw Score', 'Cosine', 'Lesk', 'F-measure')):
    """
    Run text similarity on given data and return features
    :param answer: student answer
    :param reference_answers: reference (correct) answers
    :param metrics: selects which text similarity metrics to use
    :return: an array with the averages of each selected metric
    """
    averages = [0] * len(metrics)

    for a in reference_answers:
        r = text_similarity(answer, a)
        for idx, metric in enumerate(metrics):
            if metric in r:
                averages[idx] += r[metric]

    averages = [i / len(reference_answers) for i in averages]

    return averages


def get_row_data(row):
    """
    Parse data from sample in csv dataset
    :param row: sample from dataset
    :return: answer text, question id,
    """
    answer_text = row[1].strip()  # answer text
    q_id = row[2]  # question id
    a_score = row[3]  # answer score

    # Get reference answers
    reference_answers = q_rubric_dict[q_id]

    # We are classifying based on numeric answer score, so we parse it from a_score string
    if a_score.isnumeric():
        answer_score = int(a_score)
    else:
        answer_score = cl_dict[a_score]

    return answer_text, answer_score, reference_answers


def train_classifier(csv_dataset, sklearn_classifier):
    """
    Train an sklearn classifier on the given dataset using text similarity features
    :param csv_dataset: an AnswersCSVDataset
    :param sklearn_classifier: an sklearn classifier
    :return: None
    """

    X = []
    Y = []
    for sample in tqdm(csv_dataset):
        a_text, score, references = get_row_data(sample)
        Y.append(score)

        features = get_text_features(a_text, references)
        X.append(features)

    sklearn_classifier.fit(X, Y)


def predict(csv_dataset, sklearn_classifier=None):
    """
    Test classifier on the given dataset, if no classifier is given, a simple cosine threshold will be used
    :param csv_dataset: an AnswersCSVDataset
    :param sklearn_classifier: an sklearn classifier
    :return: predictions and ground truth labels
    """
    y_true = []
    y_pred = []
    for sample in tqdm(csv_dataset):
        a_text, score, references = get_row_data(sample)
        y_true.append(score)

        # Predict using classifier
        if sklearn_classifier:
            features = get_text_features(a_text, references)
            prediction = sklearn_classifier.predict([features])
            y_pred.append(prediction[0])

        # Predict using threshold
        else:
            features = get_text_features(a_text, references, metrics=('Cosine',))

            if features[0] >= args.correct_threshold:
                y_pred.append(0)
            elif features[0] >= args.partial_threshold:
                y_pred.append(1)
            else:
                y_pred.append(2)
    return y_true, y_pred


if __name__ == '__main__':

    #
    # Arguments:
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="./answers_train.csv",
                        help='Path to the train dataset')
    parser.add_argument('--test_data_path', type=str, default="./answers_train.csv",
                        help='Path to the test dataset')

    parser.add_argument('--method', type=str, default='threshold')
    parser.add_argument('--correct_threshold', type=float, default=.2)
    parser.add_argument('--partial_threshold', type=float, default=.1)

    args = parser.parse_args()

    #
    # Load Data
    #
    test_data = AnswersCSVDataset(args.test_data_path)
    train_data = AnswersCSVDataset(args.train_data_path) if args.train_data_path else None

    #
    # Fit on train set
    #
    classifier = None

    # Decision Tree
    if args.method == 'tree':
        print(f"Training Decision Tree")
        classifier = tree.DecisionTreeClassifier()
        train_classifier(train_data, classifier)

    # Threshold
    else:
        print(f"Threshold No Training")

    # Predict
    y_true, y_pred = predict(test_data, classifier)

    #
    # Calculate Accuracy
    print(classification_report(y_true, y_pred))
