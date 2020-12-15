from collections import defaultdict
import numpy as np
from utils_pos import get_word_tag, preprocess


class Emmission_Transmission_Matrix:
    def __init__(self, training_corpus, vocab):
        self.training_data = training_corpus
        self.vocab = vocab


    def create_dictionaries(self):
        emission_counts = defaultdict(int)
        transition_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        prev_tag = '--s--'

        for word_tag in self.training_data:
            word, tag = get_word_tag(word_tag,self.vocab)

            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[tag] += 1

            prev_tag = tag

        return emission_counts, transition_counts, tag_counts

    def output_matrices(self , alpha):

        emission_counts, transition_counts, tag_counts = self.create_dictionaries()

        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)

        num_words = len(self.vocab)
        vocab_lst = list(self.vocab)

        #transition matrix initialization
        A = np.zeros((num_tags,num_tags))
        #Emission Matrix Initialization
        B = np.zeros((num_tags, num_words))


        trans_keys = set(transition_counts.keys())
        emis_keys = set(list(emission_counts.keys()))

        for i in range(num_tags):
            for j in range(num_tags):
                count = 0
                key = (all_tags[i],all_tags[j])

                if key in trans_keys:
                    count = transition_counts[key]

                count_prev_tag = tag_counts[all_tags[i]]
                A[i,j] = (count + alpha) / (count_prev_tag + alpha*num_tags)

        for i in range(num_tags):
            for j in range(num_words):
                count = 0
                key = (all_tags[i],vocab_lst[j])
                if key in emis_keys:
                    count = emission_counts[key]

                count_tag = tag_counts[all_tags[i]]
                B[i,j] = (count + alpha) / (count_tag+ alpha*num_words)

        return A , B , tag_counts
