# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:03:33 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat
from matmodel import utils, feature_builder as fb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import copy
import sys

def printf(format, *args):
    sys.stdout.write(format % args)

# LOAD DATA SET'S
def load_data(len_packet=25):
    # LABELED DATASET
    dataset = np.load('./data/morfology.npy', allow_pickle=True).item()

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



# LOAD DATA SET'S
def load_new_data(len_packet=25):
    # LABELED DATASET
    dataset = loadmat('data/morfologies_2019.mat')

    list_ecg = dataset['cell_data']['beats']
    list_peaks = dataset['cell_data']['peaks']
    list_labels = dataset['cell_data']['morfology']


    list_of_packets = []
    for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
        sig = np.subtract(sig, np.mean(sig))
        list_of_packets.append(fb.LabeledPacket(np.array(sig),
                                                peak, label))

    real_m_model = fb.ECG.build_from_packets(list_of_packets)
    list_ecg = real_m_model.build_packets(len_packet)
    list_ecg = real_m_model.packets

    return list_ecg, list_labels


def load_mix(data, data_label, new_data, new_data_label):
    # mix the real datas
    df = pd.DataFrame(data_label)
    df = df.sort_values(0)
    indexes = []
    for value, n in zip(df[0].unique(),[10, 30, 20, 60]):
        indexes.extend(df.loc[df[0]==value].index[:n])
        
    new_data_label.extend(list(np.array(data_label)[indexes]))
    new_data.extend(list(np.array(data)[indexes]))
    return new_data, new_data_label

#def load_data_modeled(len_packet, num_of_packets=20):
#    # ARTIFICIAL DATASET
#    ListName = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']
#    artificial_signals = []
#    for name in ListName:
#        for i in range(10):
#            data = utils.read_file('data/signal_' + name + '_' + str(i))
#            my_ecg = fb.deserialize(data)
#            for j in range(num_of_packets):
#                peak = np.int(len(my_ecg.packets[j])/2)
#                artificial_signals.append(fb.LabeledPacket(my_ecg.packets[j],
#                                                           peak,
#                                                           packet.labels))
#                
#
#    artificial = fb.ECG.build_from_packets(artificial_signals)
#    artificial_signals = artificial.build_packets(len_packet)
#    artificial_signals = artificial.packets
#    labels = []
#    for packet in artificial_signals:
#        labels.append(packet.label)
#    return artificial_signals, labels


def load_data_modeled(len_packet, num_of_packets=20):
    # ARTIFICIAL DATASET
    ListName = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']
    artificial_t = []
    artificial_signals = []
    labels = []
    for name in ListName:
        for i in range(10):
            data = utils.read_file('data/signal_' + name + '_' + str(i))
            my_ecg = fb.deserialize(data)
            for j in range(num_of_packets):
                artificial_t.append(name)
                artificial_signals.append(my_ecg.packets[j])
                labels.append(my_ecg.packets[j].label)

    artificial = fb.ECG.build_from_packets(artificial_signals)
    artificial_signals = artificial.build_packets(len_packet)
    artificial_signals = artificial.packets
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


def generate_mathematical(list_ecg, offset, name):
    # Matemathical Models Evaluetion
    list_of_templates = utils.get_default_templates(offset)

    list_of_packets = utils.mp_signal_apply(utils.consume_packet_helper,
                                            list_ecg,
                                            list_of_templates,
                                            workers=3)
    m_model = fb.ECG.build_from_packets(list_of_packets)

    utils.write_file('m_model_' + name, m_model.serialize())


def generate_random(offset, random_matrix=None):
    random = fb.RandomExtraction(offset=offset)
    if type(random_matrix) == np.ndarray:
        random.random_matrix = random_matrix
    def my_generate_random(list_ecg, offset, name):
        random.results = []
        for sig in list_ecg:
            random.evaluete_signal(sig.input_signal)
        utils.write_file('random_model_' + name, random.serialize())
    return my_generate_random


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
def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)

def fit_ensemble_svc(X_train, X_test, y_train, y_test, number_cls,
                        nfolds=5, test_size=0.33, metric='accuracy'):

    sss = StratifiedShuffleSplit(n_splits=nfolds,
                                 test_size=test_size,
                                 random_state=np.random.randint(10000))
     # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**(i-15) for i in range(0, 19, 2)],
                        'C': [2**(i-5) for i in range(0, 21, 2)]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],
    #                      'C': [0.001, 0.01, 0.1, 1, 10]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=sss,
                       scoring=metric)

    classifiers = [clf for i in range(number_cls)]
    size_signal = np.int(X_train.shape[1]/number_cls)
    X_train_list = [X_train[:, (0 + (size_signal * (i))):(size_signal * (1 + i))] for i in range(number_cls)]
#    del X_train
    X_test_list = [X_test[:, (0 + (size_signal * (i))):(size_signal * (1 + i))] for i in range(number_cls)]
#    del X_test
    fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
    y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

    return accuracy_score(y_test, y_pred)

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
#    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]
    estimators_ = utils.mp_fit_cls(utils.consume_fit_cls, classifiers, X_list, y)
#    estimators_ = [clf.fit(X, y) for clf, X in zip(classifiers, X_list)]

    return estimators_, le_


def svc_param_selection(X_train, X_test, y_train, y_test,
                        nfolds=5, test_size=0.33, metric='accuracy'):
    r = np.random.randint(10000)
    print(r)
    sss = StratifiedShuffleSplit(n_splits=nfolds,
                                 test_size=test_size,
                                 random_state=r)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**(i-15) for i in range(0, 19, 2)],
                         'C': [2**(i-5) for i in range(0, 21, 2)]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=sss,
                       scoring=metric)
    clf.fit(X_train, y_train)
#    y_true, y_pred = y_test, clf.predict(X_test)
    return clf.score(X_test, y_test)
#    return accuracy_score(y_true, y_pred)

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

##########################Check sanity##########################################33
def evaluete_evaluete_mathematical_sanity(target):
    X = load_mathematical('m_model_0')
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    return evaluete_model(X_train,
                          X_test,
                          y_train,
                          y_test)

def evaluete_wavelet_sanity(target):
    model_train = restore_model('wavelet_model_0')
    X = model_train.get_table_erros().values[:, 2:10]
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    return evaluete_model(X_train,
                          X_test,
                          y_train,
                          y_test)
    
    
def evaluete_evaluete_random_sanity(target):
    model_train = restore_model('random_model_0')
    X = model_train.get_table_erros().values
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    return evaluete_model(X_train,
                          X_test,
                          y_train,
                          y_test)
    

def evaluete_evaluete_morfology_sanity(target):
    model_train = restore_model('morfology_model_0')
    X = model_train.get_table_erros().values
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    return evaluete_model(X_train,
                          X_test,
                          y_train,
                          y_test)
####################################################################

def aux_load_mathematical(name):
    model = restore_model(name)
    table = model.get_errors_table(normalized=False)
    table = MinMaxScaler().fit_transform(table.T)
    table = pd.DataFrame(table.T)
    table['is_valey'] = [int(x.is_valey) for x in model.packets]
    return table


def load_mathematical(s_type, noise=0):
    if s_type == 'real_test':
        return aux_load_mathematical('m_model_' + str(0) + '_real3')
    elif s_type == 'real_train':
        return aux_load_mathematical('m_model_' + str(0) + '_real2')
    else:
        return aux_load_mathematical('m_model_' + str(noise))


def load_random(s_type, noise=0):
    if s_type == 'real_test':
        return restore_model('random_model_' + str(0) + '_real3').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('random_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('random_model_' + str(noise)).get_table_erros().values


def load_wavelet(s_type, noise=0):
    if s_type == 'real_test':
        return restore_model('wavelet_model_' + str(0) + '_real3').get_table_erros().values[:, 4:10]
    elif s_type == 'real_train':
        return restore_model('wavelet_model_' + str(0) + '_real2').get_table_erros().values[:, 4:10]
    else:
        return restore_model('wavelet_model_' + str(noise)).get_table_erros().values[:, 4:10]


def load_morfology(s_type, noise=0):
    if s_type == 'real_test':
        return restore_model('morfology_model_' + str(0) + '_real3').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('morfology_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('morfology_model_' + str(noise)).get_table_erros().values



def evaluate_methodology(func_load, func_evaluate, noises, target, target2,
                         target_test, training_real=False):
    results = []
    x_test = func_load('real_test')

    for noise in noises:
        x_train = func_load('artificial', noise)
        if training_real:
            x_train_real = func_load('real_train')
            x_train = np.concatenate([x_train, x_train_real])
            input_target = target.copy()
            input_target.extend(target2)
        else:
            input_target = target
        results.append(func_evaluate(x_train,
                                     x_test,
                                     input_target,
                                     target_test))
    return results
################Func Evaluation####################333
def evaluete_model_ensemble(number_cls):
    
    def aux_evaluete_model_ensemble(X_train, X_test, y_train, y_test):
        result = []
        for i in range(5):
                result.append(fit_ensemble_svc(X_train, X_test,
                                               y_train, y_test,
                                               number_cls))
        return result
    return aux_evaluete_model_ensemble

def evaluete_model(X_train, X_test, y_train, y_test, n=5):
    result = []
    for i in range(n):
            result.append(svc_param_selection(X_train, X_test,
                                              y_train, y_test))
    return result
################## Aux Functions##########################33
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

def print_status(evaluetions):
    printf("Acurracy:%.3f(+-%.4f)\n", np.mean(evaluetions), np.std(evaluetions))

##################Main function#########################3

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    offset = 35
    freq = 360
    noises = [0, 0.1, 0.2, 0.3]
    smooth_signal = True
    number_cls = 15
    data, data_label = load_data(offset)
    data_modeled, data_modeled_label = load_data_modeled(offset, 25)
    new_data, new_data_label = load_new_data(offset)
    mix_data, mix_label_data = load_mix(data, data_label, new_data, new_data_label)


    random_model = restore_model('random_model_' + str(0) + '_real')
    func_random = generate_random(offset, random_model.random_matrix)
    func_eval_random = evaluete_model_ensemble(number_cls)

    # create mix data

    # create_models(mix_data, generate_morfology, offset, [0], '_real3')
    # create_models(mix_data, generate_wavelet, freq, [0], '_real3')
    # create_models(mix_data, generate_mathematical, offset, [0], '_real3')
    # create_models(mix_data, func_random, offset, [0], '_real3')
    


    # morfo_xp3 = evaluate_methodology(load_morfology, evaluete_model,
    #                                   noises, data_modeled_label,
    #                                   new_data_label, mix_label_data, False)

    # wave_xp3 = evaluate_methodology(load_wavelet, evaluete_model,
    #                                   noises, data_modeled_label,
    #                                   new_data_label, mix_label_data, False)

    # mathe_xp3 = evaluate_methodology(load_mathematical, evaluete_model,
    #                                   noises, data_modeled_label,
    #                                   new_data_label, mix_label_data, False)

    random_xp3 = evaluate_methodology(load_random, func_eval_random,
                                      noises, data_modeled_label,
                                      new_data_label, mix_label_data, False)

    # print("Morfologie")
    # for x in  morfo_xp3: print_status(x) 
    # print("Wavelets")
    # for x in  wave_xp3: print_status(x) 
    print("Random")
    for x in  random_xp3: print_status(x) 
    # print("Matemathical")
    # for x in  mathe_xp3: print_status(x) 


