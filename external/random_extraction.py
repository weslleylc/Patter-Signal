# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal


class RandomExtraction():
    def __init__(self, n_classifiers=15, offset=50, n_features=50):
        self.n_classifiers = n_classifiers
        self.n_features = n_features
        self.offset = offset
        self.results = []
        self.random_matrix = np.random.randn(2 * self.offset *
                                             self.n_classifiers,
                                             self.n_features)

    def evaluete_signal(self, input_signal):

        if len(input_signal) != 2 * self.offset:
            input_signal = signal.resample(input_signal, 2 * self.offset)
        results = []
        for i in range(self.n_classifiers):
            x = np.matmul(input_signal, self.random_matrix[(0 + (2 *
                          self.offset * (i))):(2 * self.offset * (1 + i)), :])
            results = np.concatenate((results, x))

        self.results.append(results)

    def get_table_erros(self):
        return pd.DataFrame(data=self.results)

    def args():
