from torch.utils.data import Dataset
import csv


class QuestionsCSVDataset(Dataset):
    """
    Minimalistic Dataset
        - No tokenization
        - No padding
        - No class dictionary
    """

    def __init__(self, answer_file_path):

        self.data_list, self.answer_ids, self.header = self.load_data(answer_file_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def load_data(self, data_file):
        data_list = []
        answer_ids = set()
        header = None

        for file in data_file:
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)
                for row in csv_reader:
                    a_id = row[0]  # answer id

                    if a_id not in answer_ids:
                        answer_ids.add(a_id)

                    line = row[1]  # answer text
                    q_id = row[2]  # question id
                    a_score = row[3]  # answer score
                    a_feed = row[4]  # answer feedback
                    data_list.append((a_id, line, q_id, a_score, a_feed))

        return data_list, answer_ids, header


class AnswersCSVDataset(Dataset):
    """
    Minimalistic Dataset
        - No tokenization
        - No padding
        - No class dictionary
    """

    def __init__(self, answer_file_path):

        self.data_list, self.question_dict, self.header = self.load_data(answer_file_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def load_data(self, data_file):
        data_list = []
        question_dict = {}
        header = None

        for file in data_file:
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)
                for row in csv_reader:
                    q_id = row[0]  # question id
                    q_text = row[1]
                    question_dict[q_id] = q_text
                    data_list.append((q_id, q_text))

        return data_list, question_dict, header