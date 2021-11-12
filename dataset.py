from torch.utils.data import Dataset
import csv


class AnswersCSVDataset(Dataset):
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


q_rubric_dict = {'q2_a': ["Student provides rationale with accommodation for variability (e.g. repeat test method "
                          "many times; compare to chance model)", "Student describes analysis of "
                                                                  "probability/proportion/number of "
                                                                  "correct-incorrect", "Student advocates use of "
                                                                                       "statistical inference",
                          "We should use statistical inference because there could be cases of Carla just getting "
                          "lucky if we just count how many times she gets a note right.",
                          "Student advocates use of statistical inference AND Student provides rationale with "
                          "accommodation for variability (e.g. repeat test method many times)",
                          "It is sufficient that the student only implies repeating the test method many times ("
                          "i.e.'count how many times she gets a note right' in this case)",
                          "You could use statistical inference to determine if Carla has a good by comparing the "
                          "distribution of her results to those of others.",
                          "Student advocates use of statistical inference AND Student provides rationale with "
                          "accommodation for variability (e.g. repeat test method many times)",
                          "It is sufficient that the student only implies repeating the test method many times (i.e. "
                          "'distribution of her results' in this case) "
                          ],
               'q2_b': ["Student names OR paraphrases a binomial procedure, one-proportion Z-procedure, or analogous "
                        "inferential method (e.g. inferential analysis of probability/proportion/number of "
                        "correct-incorrect)",
                        "There is a 1/7 chance that a student will guess the correct note. I would then take the "
                        "ammount of times she guessed correctly (observed) compared to the number I expected her to "
                        "guess (1/7). Once I have her observed count I can run a chi square test to determine if she "
                        "was really guessing or has a good ear for music by looking at the p value.",
                        "We could run a number of note identification tests and then construct a 95% confidence "
                        "interval stating the percentage of notes she gets right. "
                        ],
               'q3_a': ["Student does NOT advocate for statistical inference (i.e. 'No'), AND recognizes that the "
                        "engineer/company has access to complete information about the order.",
                        "Student describes that decision is made directly from the number/proportion of bad displays",
                        "Student describes that engineer has access to entire population of interest",
                        "Student describes that there is no sampling",
                        "Personally, I would not use statistical inference in this scenario. Since the company must "
                        "accept or reject the entire bulk, I would make sure that less than 5% of the screens are bad "
                        "by testing all of them. If you rely on a statistic then there could sometimes be more and "
                        "sometimes less than the 5% but I would not risk it in this case.",
                        "You should not use statistical inference for this because it would be easier to just count. "
                        "If over 7 screens are bad, the whole lot can be rejected.",
                        "no because statistical inference is the theory, methods, and practice of forming judgments "
                        "about the parameters of a population, usually on the basis of random sampling "
                        ],
               'q3_b': ["Student recommends using the absolute threshold based on engineering data (5% OR 7.5 "
                        "screens; 7 or 8 are both accepted)",
                        "If there are over 7 bad screens, they should reject.",
                        "count how many out of 150 are damaged then divide it by 150. if it is greater than .05 then "
                        "reject it "
                        ],
               'q4_a': ["Student advocates use of statistical inference",
                        "Student provides rationale with accommodation for sampling variability (e.g. compare to chance model)",
                        "Student describes analysis of the average (or median) size/length/weight of fish caught byeach brother",
                        "Yes, statistical inference should be used in this scenario. We want to know if the averages for Mark and Dan differ too much to happen just by chance."
                        ],
               'q4_b': ["Student names appropriate inferential statistical method (e.g. two-sample t-procedure, oranalogous)",
                        "Student describes inferential statistical analysis of the average (e.g. mean, median) size/length/weight of fish caught by each brother",
                        "The mean fish lengths and variance of fish lengths for each fisherman's catch can be calculated. A two sample t-test can then be used to determine whether there exists a significant difference in mean lengths between the two samples. If there exists a significant difference in mean lengths, the better fisherman can be determined.",
                        "We would conduct a hypothesis test to see if Mark's average length is larger than Dan's."
                        ]
}