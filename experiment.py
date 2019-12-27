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
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
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
        
    mix_data = new_data.copy()
    mix_data_label = new_data_label.copy()
    
    mix_data_label.extend(list(np.array(data_label)[indexes]))
    mix_data.extend(list(np.array(data)[indexes]))
    return mix_data, mix_data_label


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
def generate_morfology(list_ecg, parameters, name):
    morfology = fb.MorfologyDetector(a=parameters[:2], b=parameters[2:4], c=parameters[4:6], d=parameters[6:])
    for sig in list_ecg:
        morfology.evaluete_signal(sig.input_signal,
                                  int(len(sig.input_signal)/2)+1,
                                  is_valey=sig.is_valey, label=sig.label)

    utils.write_file('models/morfology_model_' + name, morfology.serialize())


def generate_lbp(list_ecg, freq, name):
    lbp = fb.LBP1DExtraction()
    for sig in list_ecg:
        lbp.evaluete_signal(sig.input_signal, label=sig.label)

    utils.write_file('models/lbp_model_' + name, lbp.serialize())

def generate_ulbp(list_ecg, freq, name):
    ulbp = fb.ULBP1DExtraction()
    for sig in list_ecg:
        ulbp.evaluete_signal(sig.input_signal, label=sig.label)

    utils.write_file('models/ulbp_model_' + name, ulbp.serialize())

def generate_hos(list_ecg, freq, name):
    hos = fb.HOSExtraction()
    for sig in list_ecg:
        hos.evaluete_signal(sig.input_signal, label=sig.label)

    utils.write_file('models/hos_model_' + name, hos.serialize())

def generate_wavelet_db1(list_ecg, freq, name):
    wavelet_db1 = fb.WaveletTransform()
    for sig in list_ecg:
        wavelet_db1.evaluete_signal(sig.input_signal, label=sig.label)

    utils.write_file('models/wavelet_db1_model_' + name, wavelet_db1.serialize())


def generate_hermite(list_ecg, freq, name):
    hermite = fb.HermiteExtraction()
    for sig in list_ecg:
        hermite.evaluete_signal(sig.input_signal, label=sig.label)

    utils.write_file('models/hermite_model_' + name, hermite.serialize())

def generate_wavelet(list_ecg, parameters, name):
    freq = parameters[0]
    epi_qrs = [parameters[1] for i in range(5)]
    wavelet = fb.WaveletExtraction(epi_qrs=epi_qrs)
    for sig in list_ecg:
        wavelet.evaluete_signal(sig.input_signal, freq, label=sig.label)
    utils.write_file('models/wavelet_model_' + name, wavelet.serialize())


def generate_mathematical(list_ecg, offset, name):
    # Matemathical Models Evaluetion
    list_of_templates = utils.get_default_templates(offset)

    list_of_packets = utils.mp_signal_apply(utils.consume_packet_helper,
                                            list_ecg,
                                            list_of_templates,
                                            workers=3)
    m_model = fb.ECG.build_from_packets(list_of_packets)

    utils.write_file('models/m_model_' + name, m_model.serialize())


def generate_random(offset, random_matrix=None):
    random = fb.RandomExtraction(offset=offset)
    if type(random_matrix) == np.ndarray:
        random.random_matrix = random_matrix
    def my_generate_random(list_ecg, offset, name):
        random.results = []
        for sig in list_ecg:
            random.evaluete_signal(sig.input_signal, label=sig.label)
        utils.write_file('models/random_model_' + name, random.serialize())
    return my_generate_random

def restore_model(name):
    return fb.deserialize(utils.read_file(name))

def default_load(name):
    return fb.deserialize(utils.read_file(name)).get_table_erros().values

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

def fit_ensemble_svc(X_train, X_test, y_train, y_test, number_cls=15,
                    r=42, nfolds=5, test_size=0.2, metric='accuracy'):

    sss = StratifiedKFold(n_splits=nfolds,
                          random_state=r)
     # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**(i-15) for i in range(0, 19, 2)],
                        'C': [2**(i-5) for i in range(0, 21, 2)]},
                        {'kernel': ['linear'], 'C': [2**(i-5) for i in range(0, 21, 2)]}]

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


def svc_param_selection(X_train, X_test, y_train, y_test, r=42,
                        nfolds=5, test_size=0.2, metric='accuracy'):

    sss = StratifiedKFold(n_splits=nfolds,
                          random_state=r)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**(i-15) for i in range(0, 19, 2)],
                         'C': [2**(i-5) for i in range(0, 21, 2)]},
                        {'kernel': ['linear'], 'C': [2**(i-5) for i in range(0, 21, 2)]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=sss, scoring=metric)
    clf.fit(X_train, y_train)
#    y_true, y_pred = y_test, clf.predict(X_test)
    return clf.score(X_test, y_test)
#    return accuracy_score(y_true, y_pred)

def create_models(packets, func_evaluete, parameter,
                  noises=[0, 0.05, 0.15, 0.2],
                  name='', smooth_signal=False):

    for noise in noises:
        clone_packets = copy.deepcopy(packets)
        if noise != 0:
            add_noise_packets(clone_packets, noise)
        if smooth_signal:
            remove_noise_packets(clone_packets)
        for p in parameter:
                func_evaluete(clone_packets, p, str(noise) + "".join([str(x) for x in p]) + name)


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
def evaluete_sanity(name, func_eval, random_vector, func_load=default_load, n=10):
    X = func_load(name)
    target = X[:, -1]
    X = X[:, :-1]
    results = []
    # 10 signals for 4 morfologies
    size = int(len(X)/40)
    groups = [np.ones(size) * i for i in range(40)]
    groups = np.concatenate(groups)
    gss = GroupShuffleSplit(n_splits=n, train_size=.66, random_state=42)
    for i, (train_idx, test_idx) in enumerate(gss.split(X, target, groups)):
        result = func_eval(X[train_idx, :], X[test_idx, :], target[train_idx], target[test_idx])
        results.append(result)
    return results

def evaluete_sanity_with_parameters(name, func_eval, parameters, random_vector, func_load=default_load, n=10):
    results = []
    for p in parameters:
        X = func_load(name + "".join([str(x) for x in p]))
        target = X[:, -1]
        X = X[:, :-1]

        # 10 signals for 4 morfologies
        size = int(len(X) / 40)
        groups = [np.ones(size) * i for i in range(40)]
        groups = np.concatenate(groups)
        gss = GroupShuffleSplit(n_splits=n, train_size=.66, random_state=42)
        parcial_results = []
        for i, (train_idx, test_idx) in enumerate(gss.split(X, target, groups)):
            result = func_eval(X[train_idx, :], X[test_idx, :], target[train_idx], target[test_idx], 42)
            parcial_results.append(result)
        results.append(parcial_results)
    mean_results = [np.mean(x) for x in results]
    idx = np.argmax(mean_results)
    return parameters[idx], results[idx]


####################################################################


def aux_load_mathematical(name):
    model = restore_model(name)
    table = model.get_errors_table(normalized=False)
    table = MinMaxScaler().fit_transform(table.T)
    table = pd.DataFrame(table.T)
    table['is_valey'] = [int(x.is_valey) for x in model.packets]
    table['label'] = model.labels
    return table.values


def load_mathematical(s_type, noise=0):
    if s_type == 'real_test':
        return aux_load_mathematical('models/m_model_' + str(0) + '_real')
    elif s_type == 'real_train':
        return aux_load_mathematical('models/m_model_' + str(0) + '_real2')
    else:
        return aux_load_mathematical('models/m_model_' + str(noise))


def load_random(s_type, noise=0):
    if s_type == 'real_test':
        return restore_model('models/random_model_' + str(0) + '_real').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('models/random_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('models/random_model_' + str(noise)).get_table_erros().values


def load_wavelet(s_type, noise=0):
    if s_type == 'real_test':
        return restore_model('models/wavelet_model_' + str(0) + '_real').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('models/wavelet_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('models/wavelet_model_' + str(noise)).get_table_erros().values


def load_morfology(s_type, noise=0):
    if s_type == 'real_test':
        restore_model('models/morfology_model_' + str(0) + '_real').get_table_erros().values
        return restore_model('models/morfology_model_' + str(0) + '_real').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('models/morfology_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('models/morfology_model_' + str(noise)).get_table_erros().values


def load_hos(s_type, noise=0):
    if s_type == 'real_test':
        restore_model('models/hos_model_' + str(0) + '_real').get_table_erros().values
        return restore_model('models/hos_model_' + str(0) + '_real').get_table_erros().values
    elif s_type == 'real_train':
        return restore_model('models/hos_model_' + str(0) + '_real2').get_table_erros().values
    else:
        return restore_model('models/hos_model_' + str(noise)).get_table_erros().values

def load_generic(name, second_name="",load_function=restore_model):
    def aux_load_generict(s_type, noise=0):
        if s_type == 'real_test':
            return load_function('models/{}_model_{}{}_real'.format(name, noise, second_name)).get_table_erros().values
        elif s_type == 'real_train':
            return load_function('models/{}_model_{}{}_real2'.format(name, noise, second_name)).get_table_erros().values
        else:
            return load_function('models/{}_model_{}{}'.format(name, noise, second_name)).get_table_erros().values
    return aux_load_generict


def evaluate_methodology(func_load, func_evaluate, noises, random_vector):
    results = []
    data = func_load('real_test')
    x_test, target_test = data[:, :-1], data[:, -1]
    del data
    for noise in noises:
        data = func_load('artificial', noise)
        x_train, target = data[:, :-1], data[:, -1]
        del data
        results.append(func_evaluate(x_train,
                                     x_test,
                                     target,
                                     target_test,
                                     random_vector))
    return results

################Func Evaluation####################333
def evaluete_model_ensemble(number_cls=15):
    
    def aux_evaluete_model_ensemble(X_train, X_test, y_train, y_test, random_vector):
        result = []
        for i in range(10):
                result.append(fit_ensemble_svc(X_train, X_test,
                                               y_train, y_test,
                                               number_cls,
                                               random_vector[i]))
        return result
    return aux_evaluete_model_ensemble

def evaluete_model(X_train, X_test, y_train, y_test, random_vector, n=10):
    result = []

    for i in range(n):
            result.append(svc_param_selection(X_train, X_test,
                                              y_train, y_test, random_vector[i]))
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

##################Main function#########################

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # window of size
    offset = 35
    freq = 360
    noises = [0, 0.1, 0.2, 0.3]
    smooth_signal = False
    # random projection
    number_cls = 15
    data, data_label = load_data(offset)
    data_modeled, data_modeled_label = load_data_modeled(offset, 25)
    new_data, new_data_label = load_new_data(offset)
    mix_data, mix_label_data = load_mix(data, data_label, new_data, new_data_label.copy())
    # random_vector = [np.random.randint(10000) for i in range(10)]
    # for reprodutibility
    random_vector = [2133, 7724, 3448, 7618, 9081, 8414, 4410, 1045, 9933, 5423]

    func_random = generate_random(offset)
    # random_model = restore_model('models/random_model_' + str(0) + '_real')
    # func_random = generate_random(offset, random_model.random_matrix)
    func_eval_random = evaluete_model_ensemble(number_cls)

    #hyperparameters
    #wavelets
    wavelets_parameters = [[freq, x] for x in np.linspace(0.4, 0.6, 5)]
    morfology_parameters = [[0, 40, x1, x1 + 10, x2, x2 + 10, 150, 180] for x1, x2 in zip(range(45, 76, 10), range(125, 94, -10))]
    morfology_parameters2 = [[0, 40, x1, x1 + 20, x2, x2 + 10, 150, 180] for x1, x2 in zip(range(45, 66, 10), range(115, 94, -10))]
    morfology_parameters3 = [[0, 40, x1, x1 + 30, x2, x2 + 10, 150, 180] for x1, x2 in zip(range(45, 56, 10), range(105, 94, -10))]
    morfology_parameters4 = [[0, 40, 45, 85, 95, 135, 150, 180]]
    morfology_parameters.extend(morfology_parameters2)
    morfology_parameters.extend(morfology_parameters3)
    morfology_parameters.extend(morfology_parameters4)

    # create artificial data
    # create_models(data_modeled, generate_morfology, morfology_parameters, noises, smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_wavelet, wavelets_parameters, noises, smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_mathematical, [offset], noises, smooth_signal=smooth_signal)
    # create_models(data_modeled, func_random, [[offset]], noises,smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_lbp, [offset], noises,smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_ulbp, [offset], noises,smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_hos, [offset], noises,smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_wavelet_db1, [offset], noises,smooth_signal=smooth_signal)
    # create_models(data_modeled, generate_hermite, [offset], noises, smooth_signal=smooth_signal)


    # selected_morfo, morfo_xp1 = evaluete_sanity_with_parameters('models/morfology_model_0',
    #                                                             parameters=morfology_parameters,
    #                                                             random_vector=random_vector,
    #                                                             func_eval=svc_param_selection)
    # selected_wavelet, wave_xp1 = evaluete_sanity_with_parameters('models/wavelet_model_0',
    #                                                             parameters=wavelets_parameters,
    #                                                             random_vector=random_vector,
    #                                                             func_eval=svc_param_selection)

    # random_xp1 = evaluete_sanity('models/random_model_0', random_vector=random_vector,
    #                              func_eval=fit_ensemble_svc)
    # mathe_xp1 = evaluete_sanity('models/m_model_0', func_eval=svc_param_selection, random_vector=random_vector, func_load=load_mathematical)
    # lbp_xp1 = evaluete_sanity('models/lbp_model_0', random_vector=random_vector, func_eval=svc_param_selection)
    # ulbp_xp1 = evaluete_sanity('models/ulbp_model_0', random_vector=random_vector, func_eval=svc_param_selection)
    # hos_xp1 = evaluete_sanity('models/hos_model_0', random_vector=random_vector, func_eval=svc_param_selection)
    # wavelet_db1_xp1 = evaluete_sanity('models/wavelet_db1_model_0', random_vector=random_vector, func_eval=svc_param_selection)
    # hermite_xp1 = evaluete_sanity('models/hermite_model_0', random_vector=random_vector, func_eval=svc_param_selection)
    #
    # print("Wavelets")
    # print_status(wave_xp1)
    # print("Morfologie")
    # print_status(morfo_xp1)
    # print("Random")
    # print_status(random_xp1)
    # print("Matemathical")
    # print_status(mathe_xp1)
    # print("lbp")
    # print_status(lbp_xp1)
    # print("ulbp")
    # print_status(ulbp_xp1)
    # print("hos")
    # print_status(hos_xp1)
    # print("wavelet_db1")
    # print_status(wavelet_db1_xp1)
    # print("hermite")
    # print_status(hermite_xp1)

    # create mix data
    # create_models(mix_data, generate_morfology, [selected_morfo], [0], '_real', smooth_signal=smooth_signal)
    # create_models(mix_data, generate_wavelet, [selected_wavelet], [0], '_real', smooth_signal=smooth_signal)
    # create_models(mix_data, generate_mathematical, [[offset]], [0], '_real',smooth_signal=smooth_signal)
    # create_models(mix_data, func_random, [[offset]], [0], '_real', smooth_signal=smooth_signal)
    # create_models(mix_data, generate_lbp, [[offset]], [0], '_real',smooth_signal=smooth_signal)
    # create_models(mix_data, generate_ulbp, [[offset]], [0], '_real', smooth_signal=smooth_signal)
    # create_models(mix_data, generate_hos, [[offset]], [0], '_real',smooth_signal=smooth_signal)
    # create_models(mix_data, generate_wavelet_db1, [[offset]], [0], '_real',smooth_signal=smooth_signal)
    # create_models(mix_data, generate_hermite, [[offset]], [0], '_real',smooth_signal=smooth_signal)

    # morfo_xp2 = evaluate_methodology(load_generic('morfology', "".join([str(x) for x in selected_morfo])),
    #                                  evaluete_model, noises, random_vector)
    # wave_xp2 = evaluate_methodology(load_generic('wavelet', "".join([str(x) for x in selected_wavelet])),
    #                                 evaluete_model, noises, random_vector)
    mathe_xp2 = evaluate_methodology(load_mathematical, evaluete_model, noises, random_vector)
    # random_xp2 = evaluate_methodology(load_generic('random'), func_eval_random, noises, random_vector)
    # lbp_xp2 = evaluate_methodology(load_generic('lbp'), evaluete_model, noises, random_vector)
    # hos_xp2 = evaluate_methodology(load_generic('hos'), evaluete_model, noises, random_vector)
    # ulbp_xp2 = evaluate_methodology(load_generic('ulbp'), evaluete_model, noises, random_vector)
    # wavelet_db1_xp2 = evaluate_methodology(load_generic('wavelet_db1'), evaluete_model, noises, random_vector)
    hermite_xp2 = evaluate_methodology(load_generic('hermite'), evaluete_model, noises, random_vector)


    # print("Morfologie")
    # for x in  morfo_xp2: print_status(x)
    # print("Wavelets")
    # for x in  wave_xp2: print_status(x)
    # print("Random")
    # for x in  random_xp2: print_status(x)
    # print("lbp")
    # for x in lbp_xp2: print_status(x)
    # print("ulbp")
    # for x in ulbp_xp2: print_status(x)
    # print("hos")
    # for x in hos_xp2: print_status(x)
    # print("wavelet_db1")
    # for x in wavelet_db1_xp2: print_status(x)
    print("Matemathical")
    for x in mathe_xp2: print_status(x)
    print("hermite")
    for x in hermite_xp2: print_status(x)




