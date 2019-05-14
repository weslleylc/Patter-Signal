# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:03:33 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matmodel import utils, feature_builder as fb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from scipy import stats
import copy


# LOAD DATA SET'S
def load_data(len_packet=25):
    # LABELED DATASET
    dataset = np.load('./data/morfology.npy').item()

    list_ecg = dataset['signal']
    list_peaks = dataset['peaks']
    list_labels = dataset['label']

    list_of_packets = []
    for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
        sig = np.subtract(sig, np.mean(sig))
        if label == 2:
            label = 'qRs_4'
        if label == 14:
            label = 'QS_5'
        if label == 10:
            label = 'rRs_8'
        if label == 5:
            label = 'RS_9'
        if label not in [12, 11, 1, 18, 6, 15, -1]:
            list_of_packets.append(fb.LabeledPacket(np.array(sig),
                                                    peak, label))

    real_m_model = fb.ECG.build_from_packets(list_of_packets)
    list_ecg = real_m_model.build_packets(len_packet)
    list_ecg = real_m_model.packets
    labels = []
    for packet in list_ecg:
        labels.append(packet.label)
    return list_ecg, labels


def load_data_modeled(num_of_packets=20):
    # ARTIFICIAL DATASET
    ListName = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']
    artificial_t = []
    artificial_signals = []
    for name in ListName:
        for i in range(10):
            data = utils.read_file('data/signal_' + name + '_' + str(i))
            my_ecg = fb.deserialize(data)
            for j in range(num_of_packets):
                artificial_t.append(name)
                artificial_signals.append(my_ecg.packets[j])

    labels = []
    for packet in artificial_signals:
        labels.append(packet.label)
    return artificial_signals, labels


# EXTRACT FEATURES FOR EACH METHODOLOGY
def generate_morfology(list_ecg, freq, name):
    morfology = fb.MorfologyDetector()
    for sig in list_ecg:
        morfology.evaluete_signal(sig.input_signal,
                                  int(len(sig.input_signal)/2)+1)

    utils.write_file('morfology_model_' + name, morfology.serialize())


def generate_wavelet(list_ecg, freq, name):
    wavelet = fb.WaveletExtraction()
    for sig in list_ecg:
        wavelet.evaluete_signal(sig.input_signal, freq)

    utils.write_file('wavelet_model_' + name, wavelet.serialize())


def generate_matemhatical(list_ecg, offset, name):
    # Matemathical Models Evaluetion
    list_of_templates = utils.get_default_templates(offset)

    list_of_packets = utils.mp_signal_apply(utils.consume_packet_helper,
                                            list_ecg,
                                            list_of_templates,
                                            workers=3)
    m_model = fb.ECG.build_from_packets(list_of_packets)

    utils.write_file('m_model_' + name, m_model.serialize())


def generate_random(list_ecg, offset, name):
    random = fb.RandomExtraction(offset=offset)
    for sig in list_ecg:
        random.evaluete_signal(sig.input_signal)

    utils.write_file('random_model_' + name, random.serialize())


def restore_model(name):
    return fb.deserialize(utils.read_file(name))


# SIGNAL PROCESSINS, NOISE/DENOISE
def smooth(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def remove_noise_packets(input_packets, func=smooth):
    list_of_packets = input_packets
    for packet in list_of_packets:
        input_signal = packet.input_signal
        packet.input_signal = func(input_signal)
    return list_of_packets


def add_noise_packets(input_packets, percent):
    list_of_packets = input_packets
    for packet in list_of_packets:
        input_signal = packet.input_signal
        packet.input_signal = add_noise(input_signal, percent)
    return list_of_packets


def add_noise(input_signal, percent):
    var = np.abs(rms(input_signal))
    noise = np.random.normal(0,
                             var,
                             len(input_signal))
    return np.add(input_signal, noise*percent)


def rms(x):
    return np.sqrt(np.mean(x**2))


# METHODOLOGY'S OF EVALUETION'S
def svc_param_selection(X_train, X_test, y_train, y_test,
                        nfolds=5, test_size=0.4, metric='accuracy'):

    sss = StratifiedShuffleSplit(n_splits=5,
                                 test_size=test_size,
                                 random_state=0)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=sss,
                       scoring=metric)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    return accuracy_score(y_true, y_pred)


def svc_param_selection_group(X, y, groups,
                              nfolds=5, test_size=0.4,
                              metric='accuracy'):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)
    sss = GroupKFold(n_splits=4)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'degree': [2, 3, 4, 5]}]

    clf = GridSearchCV(SVC(), tuned_parameters,
                       cv=sss.split(X_train, y_train, groups[:len(y_train)]),
                       scoring=metric)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    return accuracy_score(y_true, y_pred)


def create_models(packets, func_evaluete,
                  param, noises=[0, 0.05, 0.15, 0.2],
                  name='', smooth_signal=True):

    for noise in noises:
        clone_packets = copy.deepcopy(packets)
        add_noise_packets(clone_packets, noise)
        if smooth_signal:
            remove_noise_packets(clone_packets)
        func_evaluete(clone_packets, param, str(noise) + name)


def get_mix(x_a, y_a, x_b, y_b, x_b_raw, percent):
    if percent != 0:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=1-percent,
                                     random_state=42)
        for train_index, test_index in sss.split(x_b, y_b):
            x_a = np.concatenate((x_a, x_b_raw[train_index]))
            y_a = np.concatenate((y_a, y_b[train_index]))
            x_b = x_b[test_index]
            y_b = y_b[test_index]
    return np.array(x_a), np.array(x_b), np.array(y_a), np.array(y_b)


def evaluete_morfology(noises, target, target_test, training_noises=False):
    results = []
    table_train = []
    for noise in noises:
        if not training_noises:
            if len(table_train) == 0:
                model_train = restore_model('morfology_model_0')
                table_train = model_train.get_table_erros().values
                test_raw = restore_model('morfology_model_0_real')
                test_raw = test_raw.get_table_erros().values
        else:
            model_train = restore_model('morfology_model_' + str(noise))
            table_train = model_train.get_table_erros().values
            test_raw = restore_model('morfology_model_' + str(noise) + '_real')
            test_raw = test_raw.get_table_erros().values
        model_test = restore_model('morfology_model_' + str(noise) + '_real')
        table_test = model_test.get_table_erros().values
        results.append(evaluete_model(table_train,
                                      table_test,
                                      target,
                                      target_test,
                                      test_raw))
    return results


def evaluete_random(noises, target, target_test, training_noises=False):
    results = []
    table_train = []
    for noise in noises:
        if not training_noises:
            if len(table_train) == 0:
                model_train = restore_model('random_model_0')
                table_train = model_train.get_table_erros().values
                test_raw = restore_model('random_model_0_real')
                test_raw = test_raw.get_table_erros().values
        else:
            model_train = restore_model('random_model_' + str(noise))
            table_train = model_train.get_table_erros().values
            test_raw = restore_model('random_model_' + str(noise) + '_real')
            test_raw = test_raw.get_table_erros().values
        model_test = restore_model('random_model_' + str(noise) + '_real')
        table_test = model_test.get_table_erros().values
        results.append(evaluete_model(table_train,
                                      table_test,
                                      target,
                                      target_test,
                                      test_raw))
    return results


def evaluete_wavelet(noises, target, target_test, training_noises=False):
    results = []
    table_train = []
    for noise in noises:
        if not training_noises:
            if len(table_train) == 0:
                model_train = restore_model('wavelet_model_0')
                table_train = model_train.get_table_erros().values[:, 2:10]
                test_raw = restore_model('wavelet_model_0_real')
                test_raw = test_raw.get_table_erros().values[:, 2:10]
        else:
            model_train = restore_model('wavelet_model_' + str(noise))
            table_train = model_train.get_table_erros().values[:, 2:10]
            test_raw = restore_model('wavelet_model_' + str(noise) + '_real')
            test_raw = test_raw.get_table_erros().values[:, 2:10]
        model_test = restore_model('wavelet_model_' + str(noise) + '_real')
        table_test = model_test.get_table_erros().values[:, 2:10]
        results.append(evaluete_model(table_train,
                                      table_test,
                                      target,
                                      target_test,
                                      test_raw))
    return results


def load_mathematical(name):
    model = restore_model(name)
    table = model.get_errors_table(normalized=False)
    table = MinMaxScaler().fit_transform(table.T)
    table = pd.DataFrame(table.T)
    table['is_valey'] = [int(x.is_valey) for x in model.packets]
    return table


def evaluete_mathematical(noises, target, target_test, training_noises=False):
    results = []
    table_train = []
    for noise in noises:
        if not training_noises:
            if len(table_train) == 0:
                table_train = load_mathematical('m_model_0')
                test_raw = load_mathematical('m_model_0_real')
        else:
            table_train = load_mathematical('m_model_' + str(noise))
            test_raw = load_mathematical('m_model_' + str(noise) + '_real')

        table_test = load_mathematical('m_model_' + str(noise) + '_real')
        results.append(evaluete_model(table_train,
                                      table_test,
                                      target,
                                      target_test,
                                      test_raw))
    return results


def evaluete_model(train, test, target, target_test, test_raw):
    result = []
    for k, percent in enumerate(np.linspace(0, 0.6, 20)):
            X_train, X_test, y_train, y_test = get_mix(np.array(train),
                                                       np.array(target),
                                                       np.array(test),
                                                       np.array(target_test),
                                                       np.array(test_raw),
                                                       percent)
            result.append(svc_param_selection(X_train, X_test,
                                              y_train, y_test))
    return result

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def matrix_to_tikz(matrix , numbers, Legend):
    """
    matrix=contém os valores em formato de matriz dos graficos a serem
    printados [x]
    numbers = contém os valores em formato de matriz y
    legend=legenda
    """
    [l, c] = np.array(matrix).shape
    aux = r"\addplot"
    aux += r" table [x=n, y=Rand, col sep=comma, row sep=\\]{n, Rand\\"
    StringFinal = ''
    for i in range(l):
        string = aux
        aux2 = ''
        for j in range(c):
            aux2 += str(numbers[i][j])
            aux2 += ', '
            aux2 += truncate(matrix[i][j], 4)
            aux2 += r' \\ '
        string += aux2
        string += r'};\addlegendentry{'
        string += str(Legend[i])
        string += '}'
        StringFinal += string
    return StringFinal


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    offset = 25
    freq = 360
    noises = [0, 0.1, 0.2, 0.3, 0.4]
    data, data_label = load_data(offset)
    data_modeled, data_modeled_label = load_data()

    # create artificial data
    create_models(data_modeled, generate_morfology, noises, freq)
    create_models(data_modeled, generate_wavelet, noises, freq)
    create_models(data_modeled, generate_matemhatical, noises, offset)
    create_models(data_modeled, generate_random, noises, offset)
    # create real data
    create_models(data, generate_morfology, noises, freq, '_real')
    create_models(data, generate_wavelet, noises, freq, '_real')
    create_models(data, generate_matemhatical, noises, offset, '_real')
    create_models(data, generate_random, noises, offset, '_real')

    # evaluete models without noise on train
    morfo_xp1 = evaluete_morfology(noises,
                                   data_modeled_label,
                                   data_label,
                                   False)
    wave_xp1 = evaluete_wavelet(noises,
                                data_modeled_label,
                                data_label,
                                False)
    mathe_xp1 = evaluete_mathematical(noises,
                                      data_modeled_label,
                                      data_label,
                                      False)
    random_xp1 = evaluete_random(noises,
                                 data_modeled_label,
                                 data_label,
                                 False)

    # evaluete models with noise on train
    morfo_xp2 = evaluete_morfology(noises,
                                   data_modeled_label,
                                   data_label,
                                   True)
    wave_xp2 = evaluete_wavelet(noises,
                                data_modeled_label,
                                data_label,
                                True)
    mathe_xp2 = evaluete_mathematical(noises,
                                      data_modeled_label,
                                      data_label,
                                      True)
    random_xp2 = evaluete_random(noises,
                                 data_modeled_label,
                                 data_label,
                                 True)
