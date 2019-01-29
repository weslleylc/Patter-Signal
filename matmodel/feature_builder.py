# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:51:11 2019

@author: Weslley L. Caldas
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal as sci_signal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import multiprocessing as mp


class DefaultSignal():
    def __init__(self, input_signal, indicators=None):
        self.input_signal = input_signal
        self.indicators = indicators
        self.packets = []

    def buildPackets(self, bandwidth):
        if self.indicators[0] - bandwidth <= 0:
            raise ValueError('The bandwith must be'
                             + 'less than the first indicator')
        packets = []
        for ind in self.indicators:
            current_input_signal = self.input_signal[(
                ind-bandwidth):(ind+bandwidth)]
            packets.append(current_input_signal)
        self.packets = packets
        return self.packets


class ECG(DefaultSignal):
    def __init__(self, input_signal, indicators,
                 ts=None, filtered=None,
                 heart_rate_ts=None, heart_hate=None,
                 templates_ts=None, templates=None,
                 is_filtered=False):

        if indicators is None:
            raise ValueError('You must pass the R peaks.')

        if is_filtered:
            super().__init__(input_signal, indicators)
            self.filtered = input_signal

        else:
            super().__init__(input_signal, indicators)
            self.filtered = None
        self.ts = ts
        self.heart_rate_ts = heart_rate_ts
        self.heart_hate = heart_hate
        self.templates_ts = templates_ts
        self.templates = templates


class MathematicalModel():
    def __init__(self,
                 left_function,
                 rigth_function):

        self.left_function = left_function
        self.rigth_function = rigth_function

    def call(self, *, input_signal, left_value, left_length,
             right_value, rigth_length):

        (left_input_signal,
         rigth_input_signal) = self.split_input_signal(input_signal)

        model = self.build_model(left_value, left_length,
                                 right_value, rigth_length)

        if(type(model) != np.array):
            model = np.array(model)
        if(type(input_signal) != np.array):
            input_signal = np.array(input_signal)

        size = len(input_signal)

        left_extend = np.repeat(model[0], size)
        rigth_extend = np.repeat(model[-1], size)

        model = np.concatenate((left_extend, model, rigth_extend), axis=0)

        peak_signal = self.find_peak(input_signal)
        peak_model = self.find_peak_model(model)

        left_limit = peak_signal
        rigth_limit = size - peak_signal

        model = model[(peak_model-left_limit):(peak_model+rigth_limit)]
        (model, input_signal) = self.normalize_data(model, input_signal,
                                                    peak_signal, left_limit)
        return self.metric(input_signal, model, peak_signal)
        # return metric(model, input_signal, peak_signal)

    def split_input_signal(self, input_signal):
        """
        This function return a tuple with the left and right part of the
        input_signal, splited on their center.
        """
        idx_max = np.argmax(input_signal)
        return (input_signal[:idx_max], input_signal[idx_max:])

    def metric(self, input_signal, model, peak_signal):
        """
        This function return the evaluetion given a siginal and a mathematical
        model
        """
        left_error = mean_squared_error(input_signal[:peak_signal],
                                        model[:peak_signal])
        right_error = mean_squared_error(input_signal[peak_signal:],
                                         model[peak_signal:])
        return (left_error, right_error)

    def find_peak(self, input_signal):
        idx_max = np.argmax(input_signal)
        return idx_max

    def find_peak_model(self, input_signal):
        idx_max = np.argmax(input_signal)
        return idx_max

    def normalize_data(self, model, input_signal,
                       peak_sig, peak_model):

        normalized_input_signal = MinMaxScaler().fit_transform(
                                               input_signal.reshape(-1, 1))

        left_model = MinMaxScaler(feature_range=(
                     np.min(normalized_input_signal[:peak_sig]),
                     np.max(normalized_input_signal[:peak_sig]))
                     ).fit_transform(model[:peak_model].reshape(-1, 1))

        right_model = MinMaxScaler(feature_range=(
                     np.min(normalized_input_signal[:peak_sig]),
                     np.max(normalized_input_signal[:peak_sig]))
                     ).fit_transform(model[peak_model:].reshape(-1, 1))

        normalized_model = np.concatenate((left_model, right_model))
        return (normalized_model, normalized_input_signal)

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        r_left = math.ceil(left_length/2)
        r_rigth = math.ceil(rigth_length/2)
        left_model = self.left_function(left_length, left_value)[:-r_left]
        rigth_model = self.rigth_function(rigth_length, right_value)[-r_rigth:]
        model = np.concatenate((left_model, rigth_model), axis=0)

        return model


class MexicanHat(MathematicalModel):
    def __init__(self):
        super().__init__(sci_signal.ricker, sci_signal.ricker)


class Gaussian(MathematicalModel):
    def __init__(self):
        super().__init__(self.gaussian_func, self.gaussian_func)

    def gaussian_func(self, n, sigma):
        x = np.linspace(-n/2, n/2, n)
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp((-x**2)/2*sigma**2)


class PrintMexicanHat(MexicanHat):
    def __init__(self):
        super().__init__()

    def metric(self, input_signal, model, peak_signal):
        """
        This function return the evaluetion given a siginal and a mathematical
        model
        """
        return plt.plot(model, '-', input_signal, '-')


class Rayleigh(MathematicalModel):
    def __init__(self):
        super().__init__(self.rayleigh_func, self.rayleigh_func)

    def rayleigh_func(self, n, scale):
        x = np.linspace(0, 1, n)
        return (x/(scale**2))*np.e**((-x**2)/2*scale**2)

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)

        model = np.concatenate((left_model[::-1], rigth_model[1:]), axis=0)
        return model


class LeftInverseRayleigh(Rayleigh):
    def __init__(self):
        super().__init__()

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)

        model = np.concatenate((left_model[::-1]*-1, rigth_model[1:]), axis=0)
        return model


class RightInverseRayleigh(Rayleigh):
    def __init__(self):
        super().__init__()

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)

        model = np.concatenate((left_model[::-1], rigth_model[1:]*-1), axis=0)
        return model


class TemplateEvaluetor():
    def __init__(self, mat_model, parameter, lenght):
        self.mat_model = mat_model
        self.parameter = parameter
        self.lenght = lenght


def consume_process(args):
    print("child")
    return consume_evaluetor(args[0], args[1], args[2], args[3])


def mp_apply(function, list_of_evaluetor, input_signal, workers=2):
    with mp.Pool(processes=workers) as pool:
        result = pool.map(function, [(input_signal,
                                     list_of_evaluetor[index].mat_model,
                                     list_of_evaluetor[index].parameter,
                                     list_of_evaluetor[index].lenght)
                          for index in range(len(list_of_evaluetor))])
    pool.close()
    return [item for sublist in result for item in sublist]


def evaluete_models(function, packets_of_signals, list_of_evaluetor):
    evalueted_signals = []
    for sig in packets_of_signals:
        input_signal = sig
        result = mp_apply(function, list_of_evaluetor, input_signal, workers=2)
        # result = function((list_of_evaluetor[1], input_signal))
        evalueted_signals.append(result)
    return evalueted_signals


def consume_evaluetor(input_signal, model, parameter, lenght):
    final_left_error = np.inf
    final_right_error = np.inf

    for (left_value, right_value) in parameter:
        for (left_length, rigth_length) in lenght:
            (left_error, right_error) = model.call(input_signal=input_signal,
                                                   left_value=left_value,
                                                   left_length=left_length,
                                                   right_value=right_value,
                                                   rigth_length=rigth_length)
            if final_left_error > left_error:
                final_left_error = left_error
                final_left_value = left_value
                final_left_length = left_length

            if final_right_error > right_error:
                final_right_error = right_error
                final_right_value = right_value
                final_right_length = rigth_length

    return [final_left_error, final_left_value, final_left_length,
            final_right_error, final_right_value, final_right_length]
