# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:03:33 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matmodel import utils, feature_builder as fb
from sklearn import svm
from sklearn.model_selection import train_test_split


def main():
    dataset = np.load('./data/morfology.npy').item()
    n = 600
    list_ecg = dataset['signal'][:n]
    list_peaks = dataset['peaks'][:n]
    list_labels = dataset['label'][:n]
    list_of_packets = []
    for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
        sig == np.subtract(sig, np.mean(sig))
        list_of_packets.append(fb.LabeledPacket(np.array(sig), peak, label))

    my_ecg = fb.ECG.build_from_packets(list_of_packets)
    len_packet = 35
    list_ecg = my_ecg.build_packets(len_packet)
    list_ecg = my_ecg.packets
    return list_ecg


if __name__ == '__main__':
    list_ecg = main()
    rd = fb.RandomExtraction(offset=35)

    for sig in list_ecg:
        rd.evaluete_signal(sig.input_signal)

    utils.write_file('teste', rd.serialize())
    data = utils.read_file('teste')
    rd = fb.deserialize(data)
    table = rd.get_table_erros()
    t = []
    for p in list_ecg:
        t.append(p.label)
        
    t = ['qRs_4' if x==2 else x for x in t]
    t = ['QS_5' if x==14 else x for x in t]
    t = ['rRs_8' if x==10 else x for x in t]
    t = ['RS_9' if x==5 else x for x in t]
    t = ['artifact' if x in [12, 11, 1, 18, 6, 15, -1] else x for x in t]

    X = table.values

    X_train, X_test, y_train, y_test = train_test_split(
            X, t, test_size=0.4, random_state=0)
    
    
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

