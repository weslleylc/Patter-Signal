# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import peakutils
from external import WTdelineator as wav


class WaveletExtraction():
    def __init__(self, epi_qrs=[0.5, 0.5, 0.5, 0.5, 0.5],
                 epi_p=0.02, epi_q=0.25):
        self.epi_qrs = epi_qrs
        self.epi_p = epi_p
        self.epi_q = epi_q
        self.results = []

    def evaluete_signal(self, input_signal, fs=250):
        # Number of samples in the signal
        N = input_signal.shape[0]
        # Create the filters to apply the algorithme-a-trous
        Q = wav.waveletFilters(N, fs)
        # Perform signal decomposition
        w = wav.waveletDecomp(input_signal, Q)
        results = []
        for i, cd in enumerate(w):
            [positive, negative] = self.count_peaks(cd,
                                                    rms(cd) * self.epi_qrs[i])
            results = np.concatenate((results, [positive, negative]))
        self.results.append(results)

    def get_table_erros(self):
        return pd.DataFrame(data=self.results, columns=['w1p', 'w1n',
                                                        'w2p', 'w2n',
                                                        'w3p', 'w3n',
                                                        'w4p', 'w4n',
                                                        'w5p', 'w5n'])
#    def delineate_qrs(self, cd2, epi_qrs_on=0.05, epi_qrs_end=0.125):
#        coeffs = pywt.wavedec(input_signal, 'db1', level=4)[1:]
#
#        idx_max = peakutils.indexes(cd2, thres=epi_qrs_on)
#        max_pks = input_signal[idx_max]
#        idx_min = peakutils.indexes(-1*cd2, thres=epi_qrs_end)
#        min_pks = input_signal[idx_min]
#
#        x = np.concat([idx_max, idx_min])
#        y = np.concat([max_pks, min_pks])
#
#        n_first = np.argmin(x)
#        n_last = np.argmax(x)
#
#        left = np.abs(input_signal[:n_first])
#        right = np.abs(input_signal[:n_last])
#
#        max_modulus = np.max(np.abs(left))
#        y_qrs_pre = 0.06 * max_modulus
#
#        m_r = np.min(right)
#        while y_qrs_pre < m_r:
#            y_qrs_pre = y_qrs_pre * 1.1
#
#        max_modulus = np.max(np.abs(right))
#        y_qrs_post = 0.09 * max_modulus
#
#        m_r = np.min(left)
#        while y_qrs_post < m_r:
#            y_qrs_post = y_qrs_post * 1.1
#
#        final_left = 0
#        for i in range(0, len(left)):
#            if left[i] < y_qrs_pre:
#                break
#            final_left = i
#
#        final_right = len(right)
#        for i in range(len(right), 0):
#            if right[i] < y_qrs_post:
#                break
#            final_right = i
#        return final_left, (n_last + final_right)

    def count_peaks(self, input_signal, amp):
        if len(input_signal) >= 3:
            idx_max = peakutils.indexes(input_signal, thres=amp)
            max_pks = input_signal[idx_max]
            idx_min = peakutils.indexes(-1*input_signal, thres=amp)
            min_pks = input_signal[idx_min]
            positive = len(max_pks)
            negative = len(min_pks)
        else:
            positive = 0
            negative = 0
        return positive, negative


def rms(x):
    return np.sqrt(np.mean(x**2))
