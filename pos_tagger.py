import numpy as np
import math
class HMM:
    def __init__(self, A, B, tag_counts, vocab):
        self.A = A
        self.B = B
        self.tag_counts = tag_counts
        self.states = sorted(tag_counts.keys())
        self.vocab = vocab


    def initialize_forward(self , corpus):
        num_tags = len(self.tag_counts)

        best_probs = np.zeros((num_tags, len(corpus)))
        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

        s_idx = self.states.index("--s--")

        for i in range(num_tags):
            if self.A[s_idx,i] == 0:
                best_probs[i,0] = float('-inf')

            else:
                best_probs[i,0] = math.log(self.A[s_idx,i]) + math.log(self.B[i,self.vocab[corpus[0]]] )


        for i in range(1, len(corpus)):
            for j in range(num_tags):

                best_prob_i =  float("-inf")

                best_path_i = None

                for k in range(num_tags):
                    prob = best_probs[k,i-1]+math.log(self.A[k,j]) +math.log(self.B[j,self.vocab[corpus[i]]])
                    if prob > best_prob_i:
                        best_prob_i = prob
                        best_path_i = k

                best_probs[j,i] = best_prob_i
                best_paths[j,i] = best_path_i

        return best_probs , best_paths



    def get_tags(self,corpus):

        best_probs , best_paths = self.initialize_forward(corpus)
        m = best_paths.shape[1]
        z = [None] * m
        num_tags = best_probs.shape[0]
        best_prob_for_last_word = float('-inf')
        pred = [None] * m
        for k in range(num_tags):
            if best_probs[k,-1]>best_prob_for_last_word:
                best_prob_for_last_word = best_probs[k,-1]
                z[m - 1] = k

        pred[m - 1] = self.states[k]

        for i in range(len(corpus)-1, -1, -1):
            pos_tag_for_word_i = best_paths[np.argmax(best_probs[:,i]),i]

            z[i - 1] = best_paths[pos_tag_for_word_i,i]
            pred[i - 1] = self.states[pos_tag_for_word_i]

        return pred
