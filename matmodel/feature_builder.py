# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:51:11 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal as sci_signal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class PacktesNotInitiated(Error):
   """Raised when the pakctes are not initiated."""
   pass


"""REMOVER ESSA CLASSE"""
class TemplateEvaluetor():
    def __init__(self, mat_model, parameters):
        self.mat_model = mat_model
        self.parameters = parameters


class DefaultPacket():
    def __init__(self, input_signal, peak_signal=None, models=None,
                 errors=None, parameters=None, names=None):
        self.input_signal = input_signal
        self.peak_signal = peak_signal
        self.models = models
        self.parameters = parameters
        self.errors = errors
        self.names = names


class DefaultSignal():
    def __init__(self, input_signal, indicators=None):
        self.input_signal = input_signal
        self.indicators = indicators
        self.packets = []

    def build_packets(self, bandwidth, container):
        if self.indicators[0] - bandwidth <= 0:
            raise ValueError('The bandwith must be'
                             + 'less than the first indicator')
        packets = []
        for ind in self.indicators:
            packets.append(container(self.input_signal[(
                           ind-bandwidth):(ind+bandwidth)]))
        self.packets = packets
        return self.packets

    def get_errors_table(self, normalized=True, normalizer=MinMaxScaler()):
        if not self.packets[0].errors:
            raise PacktesNotInitiated
        else:
            list_of_arrays = []
            columns = []
            for name in self.packets[0].names:
                for i in range(len(self.packets[0].errors[0])):
                    columns.append(name + "_" + str(i))
            for packet in self.packets:
                array = np.array(packet.errors)
                array = normalizer.fit_transform(array)
                array = np.matrix.flatten(array, 'F')
                list_of_arrays.append(array)
            df = pd.DataFrame(data=np.array(list_of_arrays), columns=columns)
            return df

    def get_signals(self, normalized=True, normalizer=MinMaxScaler()):
        list_of_arrays = []

        for packet in self.packets:
            if normalized:
                list_of_arrays.append(MinMaxScaler.fit_transform(
                                      packet.input_signal))
            else:
                list_of_arrays.append(packet.input_signal)

        return list_of_arrays

    def get_signals_and_models(self, normalized=True,
                               normalizer=MinMaxScaler()):
        list_of_signals = []
        list_of_models = []
        for packet in self.packets:
            if normalized:
                list_of_signals.append(MinMaxScaler.fit_transform(
                                      packet.input_signal))
            else:
                list_of_signals.append(packet.input_signal)
            model = [item for sublist in packet.input_signal
                     for item in sublist]
            list_of_signals.append(model)

        return list_of_signals, list_of_models

    @classmethod
    def build_from_packets(cls, list_of_packets, ts=None, filtered=None,
                           heart_rate_ts=None, heart_hate=None,
                           templates_ts=None, templates=None,
                           is_filtered=False):
        list_of_signal = []
        list_of_peaks = []
        for packet in list_of_packets:
            list_of_signal.append(packet.input_signal)
            list_of_peaks.append(packet.peak_signal)
        array = np.array(list_of_signal)
        array = np.matrix.flatten(array, 'F')
        return cls(array, list_of_peaks)


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
             right_value, rigth_length, peak_signal=None):

        if not peak_signal:
            peak_signal = self.find_peak(input_signal)

        (left_input_signal,
         rigth_input_signal) = self.split_input_signal(input_signal,
                                                       peak_signal)

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
        peak_model = self.find_peak_model(model)

        left_limit = peak_signal
        rigth_limit = size - peak_signal

        left_model = model[(peak_model-left_limit):peak_model]
        right_model = model[peak_model:(peak_model+rigth_limit)]

        model = model[(peak_model-left_limit):(peak_model+rigth_limit)]
        (model, input_signal, peak_model) = self.normalize_data(model,
                                                                input_signal,
                                                                peak_signal,
                                                                left_limit)

        (left_error, right_error) = self.metric(input_signal,
                                                model,
                                                peak_signal)
        return_errors = [left_error, right_error]
        return_model = [left_model, right_model]
        return (return_model, return_errors)

    def split_input_signal(self, input_signal, peak_signal):
        """Split the input_signal into two parts.
        This function return a tuple with the left and right part of the
        input_signal, splited on their center.
        Args:
            input_signal : array [n_samples]
                A array that represents a signal
            peak_signal : int
                the index that indicate the peak of [input_signal]
        Returns:
            (left_signal, right_signal): tuple(array,array)
                A tuple that contains the calculated left and right
                parts of the signal
        Raises:
            TypeError: An error occurred accessing mean_squared_error.
        """
        return (input_signal[:peak_signal], input_signal[peak_signal:])

    def metric(self, input_signal, model, peak_signal):
        """Compute the evaluetion between input_signal and a math model.
        The metric function returns a tuple that contains the evaluation of
        a given signal and a mathematical model. The tuple contains the
        mean squared error between the left part of the model and the left part
        of the signal and the right part of the model with the right part
        of the signal. The split of left and right part's it is made split the
        signal on their peak. The division of the mathematical model it is made
        in their middle point.
        Args:
            input_signal : array [n_samples]
                A array that represents a signal
            model : array [n_samples]
                A array that represents a mathematical model
            peak_signal : int
                the index that indicate the peak of [input_signal]
        Returns:
            (left_error, right_error): tuple(float,float)
                A tuple that contains the calculated left and right
                mean squared error
        Raises:
            TypeError: An error occurred accessing mean_squared_error.
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
        return (normalized_model, normalized_input_signal, len(left_model))

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
        self.name = "mexican_hat"


class PrintMexicanHat(MexicanHat):
    def __init__(self):
        super().__init__()

    def metric(self, input_signal, model, peak_signal):

        return plt.plot(model, '-', input_signal, '-')


class Gaussian(MathematicalModel):
    def __init__(self):
        super().__init__(self.gaussian_func, self.gaussian_func)
        self.name = "gaussian"

    def gaussian_func(self, n, sigma):
        x = np.linspace(-n/2, n/2, n)
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp((-x**2)/2*sigma**2)


class Rayleigh(MathematicalModel):
    def __init__(self):
        super().__init__(self.rayleigh_func, self.rayleigh_func)
        self.name = "rayleigh"

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
        self.name = "left_inverse_rayleigh"

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)

        model = np.concatenate((left_model[::-1]*-1, rigth_model[1:]), axis=0)
        return model


class RightInverseRayleigh(Rayleigh):
    def __init__(self):
        super().__init__()
        self.name = "right_inverse_rayleigh"

    def build_model(self, left_value, left_length,
                    right_value, rigth_length):

        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)

        model = np.concatenate((left_model[::-1], rigth_model[1:]*-1), axis=0)
        return model



