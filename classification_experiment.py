# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:03:33 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
from biosppy.signals import ecg
from scipy.misc import electrocardiogram
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matmodel import utils, feature_builder as fb
from matmodel.detect_peaks import detect_peaks
from sklearn import svm
from sklearn.model_selection import train_test_split
from external import WTdelineator
def main():
    dataset = np.load('./data/morfology.npy').item()
    n = 60
    list_ecg = dataset['signal'][:n]
    list_peaks = dataset['peaks'][:n]
    list_labels = dataset['label'][:n]
    
#    # Wavelet Transform delineation
#    for s in list_ecg:
#        s = np.array(s)
#        sig = signal.resample(s, 1000)
#        Pwav, QRS, Twav = wav.signalDelineation(sig,1000)
#        if(len(QRS)>0):
#            QRS=QRS[0]
#            plt.plot(sig[QRS[0]:QRS[-1]])
#            plt.show()
#            plt.close()

    list_of_packets = []
    for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
        sig == np.subtract(sig, np.mean(sig))
        list_of_packets.append(fb.LabeledPacket(np.array(sig), peak, label))

    my_ecg = fb.ECG.build_from_packets(list_of_packets)
    len_packet = 35
    list_ecg = my_ecg.build_packets(len_packet)
    list_ecg = my_ecg.packets
#    ecg_signal = electrocardiogram()
##    ecg_signal = ecg_signal - np.mean(ecg_signal)
    (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate
     ) = ecg.ecg(signal=a, sampling_rate=960, show=False)
    
#    my_ecg = fb.ECG(input_signal=filtered, indicators=rpeaks,
#                    is_filtered=False)
#    list_ecg = my_ecg.build_packets(25, fb.DefaultPacket)
#
#    pbounds = {'a': (3, 10), 'b': (30, 100),
#               'c': (3, 10), 'd': (30, 100),
#               'e': (10, 60),'f': (10, 60)}
#
#    norm_pbounds = {'a': (0.1, 0.3), 'b': (30, 100),
#                    'c': (0.1, 0.3), 'd': (30, 100),
#                    'e': (1, 50),'f': (1, 50)}


#    parameter = dict()
#    parameter['values'] = list(zip(np.linspace(1, 10, 15),
#                               np.linspace(1, 10, 15)))
#    parameter['lenght'] = list(zip(np.linspace(len_packet*0.6, len_packet*6, 10),
#                               np.linspace(len_packet*0.6, len_packet*6, 10)))
#    parameter['cut'] = list(zip(np.linspace(0, 60, 7),
#                            np.linspace(0, 60, 7)))
#
#
#
#    l = []
#    for i in np.linspace(1, 10, 10):
#        for j in np.linspace(1, 10, 10):
#            l.append((i, j))
#
#    r_parameter = dict()
#    r_parameter['values'] = l
#    r_parameter['lenght'] = list(zip(np.linspace(len_packet*0.6, len_packet*6, 5),
#                               np.linspace(len_packet*0.6, len_packet*6, 5)))
#    r_parameter['cut'] = list(zip(np.linspace(0, 60, 7),
#                                   np.linspace(0, 60, 7)))
#
#    template_gaussian = fb.Gaussian(parameter)
#    template_mexican = fb.MexicanHat(parameter)
#    template_rayleigh = fb.Rayleigh(r_parameter)
#    template_left_rayleigh = fb.LeftInverseRayleigh(parameter)
#    template_right_rayleigh = fb.RightInverseRayleigh(parameter)
#
#    list_of_templates = [template_gaussian, template_mexican,
#                         template_rayleigh, template_left_rayleigh,
#                         template_right_rayleigh]
    list_of_templates = utils.get_default_templates(len_packet)
#    list_of_templates = [template_rayleigh]

    lst_of_packets = utils.mp_signal_apply(utils.consume_packet_helper,
                                           list_ecg, list_of_templates)

#    lst_of_packets = utils.serial_signal_apply(utils.consume_packet,
#                                           list_ecg, list_of_templates)
    my_ecg.packets = lst_of_packets
    return my_ecg


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    results = main()
    error = results.get_errors_table()

    param1 = results.get_param_table()
    [signals, models, names, errors, types] = results.get_signals_and_models(normalized=False,
                                                       only_best=True)
    data=results.serialize()
    
    utils.write_file('outfile',data)
    data=utils.read_file('outfile')

    c=fb.deserialize(data)

#    t = []
#    for p in results.packets:
#        t.append(p.label)
#        
#    error['types'] = [int(x) for x in types]
#    X = error.values
#    
#    
#    X_train, X_test, y_train, y_test = train_test_split(
#            X, t, test_size=0.4, random_state=0)
#    
#    
#    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#    clf.score(X_test, y_test)
#    for sig, mod, name, erro, t in zip(signals, models, names, errors, types):
#        if(t):
#            mod = np.multiply(mod,-1)
#            mod = np.subtract(mod,np.mean(mod))
#        sig = np.subtract(sig,np.mean(sig))
#        plt.plot(mod, 'r--',sig,'-')
#        plt.title(name + ": " + str(erro))
#        plt.show()
#        plt.close()

#    for sig, mod, list_of_names, list_of_errors, t in zip(signals, models, names, errors, types):
#        for m, n, e in zip(mod, list_of_names, list_of_errors):
#            if(t):
#                mod = np.multiply(mod,-1)
#                mod = np.subtract(mod,np.mean(mod))
#            plt.plot(m, 'r--',sig,'-')
#            plt.title(n + ": " + str(e))
#            plt.show()
#            plt.close()


