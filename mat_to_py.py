# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:03:37 2019

@author: Weslley L. Caldas
"""

from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from matmodel import utils, feature_builder as fb
import wfdb

ListName = ['QR_1', 'qR_2', 'Qr_3', 'qRs_4', 'QS_5', 'R_6', 'rR_7', 'rRs_8',
            'RS_9', 'Rs_10', 'rS_11', 'RSr_12', 'rSR_13', 'rSr_14', 'qrS_15',
            'qS_16', 'rsRs_17', 'QRs_18', 'Qrs_19', 'qrSr_20']

ListName = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']

#ListName = ['qRs_4']
for name in ListName:
    mat = loadmat('data/Morfology_' + name + '.mat')

    sample = mat['sample']
    parameters = mat['parameters']

    for i, ecg_signal in enumerate(sample):
        ecg_signal = ecg_signal[5000:-10000]
        if name == 'QS_5':
            ecg_signal = np.multiply(ecg_signal,-1)

        (ts, filtered, rpeaks, templates_ts,
         templates, heart_rate_ts, heart_rate) = ecg.ecg(signal=ecg_signal,
                                                         sampling_rate=360,
                                                         show=False)
        if name == 'QS_5':
            filtered = np.multiply(filtered,-1)
        if name == 'qRs_4':
            filtered = np.multiply(filtered,1.4)
        intervals = rpeaks[1:] - rpeaks[:-1]
        intervals = np.mean(intervals)
        heart_rate = np.mean(heart_rate)
        labels = [name for x in rpeaks]
        my_ecg = fb.ECG(input_signal=filtered, indicators=rpeaks,
                        labels=labels, is_filtered=False)
        if name == 'qRs_4':
            offset = int(intervals/7)
        elif name == 'QS_5':
            offset = int(intervals/7)
        elif name == 'rRs_8':
            offset = int(intervals/7)
        elif name == 'RS_9':
            offset = int(intervals/7)
        else:
            offset = int(intervals/7)
        offset = 40
        my_ecg.packets = my_ecg.build_packets(offset)
        utils.write_file('data/signal_' + name + '_' + str(i), my_ecg.serialize())

list_of_signals = []
for name in ListName:
    for i in range(10):
        data = utils.read_file('data/signal_' + name + '_' + str(i))
        list_of_signals.append(fb.deserialize(data))
        
for i in range(len(list_of_signals)):
    plt.plot(list_of_signals[i].packets[0].input_signal)
    plt.show()
    plt.close()