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
import warnings
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import imshow, pause
from matmodel import utils
import json

# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class PacktesNotInitiated(Error):
    """Raised when the pakctes are not initiated."""
    pass


class MissMatchError(Error):
    """Raised when the pakctes are not initiated."""
    pass


class TemplateEvaluetor():
    """REMOVER ESSA CLASSE"""
    def __init__(self, mat_model, parameters):
        self.mat_model = mat_model
        self.parameters = parameters


registry = {}

def register_class(target_class):
    registry[target_class.__name__] = target_class


def deserialize(data):
    params = json.loads(data)
    name = params['class']
    target_class = registry[name]
    return target_class(*params['args'])


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BetterSerializable(object):
    def __init__(self, *args):
        self.args = args

    def serialize(self):
        return json.dumps({
                'class': self.__class__.__name__,
                'args': self.args(),
                },
                cls=NumpyEncoder)

    def args(self):
        raise NotImplementedError("You must to implement args.")


class RegisteredSerializable(BetterSerializable, metaclass=Meta):
    pass


class DefaultPacket(RegisteredSerializable):
    def __init__(self, input_signal, peak_signal=None, models=None,
                 errors=None, parameters=None, names=None, is_valey=None,
                 remove_trends=True):
        input_signal = np.array(input_signal)
        if remove_trends:
            m = np.mean(input_signal[[0, -1]])
            input_signal = np.subtract(input_signal, m)

        self.input_signal = input_signal
        self.peak_signal = peak_signal
        self.models = models
        self.parameters = parameters
        self.errors = errors
        self.names = names
        if not is_valey:
            is_valey = self.direction_peak(input_signal)
        self.is_valey = is_valey

    def direction_peak(self, input_signal, factor=0.5):
        id_max = np.argmax(input_signal)
        id_min = np.argmin(input_signal)
        if (input_signal[id_max] < np.abs(input_signal[id_min]) * factor):
            return True
        return False

    def args(self):
        return [self.input_signal,
                self.peak_signal,
                self.label,
                self.models,
                self.errors,
                self.parameters,
                self.names]


class LabeledPacket(DefaultPacket):
    def __init__(self, input_signal, peak_signal=None, label=None,
                 models=None, errors=None, parameters=None,
                 names=None, is_valey=False, remove_trends=True):
        super().__init__(input_signal, peak_signal, models, errors,
                         parameters, names, is_valey, remove_trends)
        self.label = label

    def args(self):
        return [self.input_signal,
                self.peak_signal,
                self.label,
                self.models,
                self.errors,
                self.parameters,
                self.names,
                self.is_valey]


class DefaultSignal(RegisteredSerializable):
    def __init__(self, input_signal, indicators=None, packets=None):
        if len(input_signal) > 1:
            warnings.warn("You are passing more than one signal")

        self.input_signal = input_signal
        self.indicators = indicators
        if not packets:
            self.packets = []
        else:
            self.packets = [deserialize(x) for x in packets]


    def build_packets(self, bandwidth, container,
                      normalized=True, normalizer=MinMaxScaler()):
        if self.indicators[0] - bandwidth <= 0:
            raise ValueError('The bandwith must be'
                             + 'less than the first indicator')
        packets = []
        for ind in self.indicators:
            x = self.input_signal[(ind-bandwidth):(ind+bandwidth)]
            x = normalizer().fit_transform(x.reshape(-1, 1)).reshape(1, -1)[0]
            packets.append(container(x, bandwidth))
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
                    columns.append((name, str(i)))
            for packet in self.packets:
                array = np.array(packet.errors)
                array = np.matrix.flatten(array, 'F')
                list_of_arrays.append(array)
            df = pd.DataFrame(list_of_arrays)
            df.columns = pd.MultiIndex.from_tuples(columns)
            df.columns.name = ['Models', 'Part']

            if normalized:
                for i in range(len(df)):
                    for j in df.columns.levels[1].values:
                        df.iloc[i].loc[:, j] = normalizer.fit_transform(
                            df.iloc[i].loc[:, j].values.reshape(-1, 1)
                            ).reshape(1, -1)[0]

            return df

    def get_param_table(self, normalized=True, normalizer=MinMaxScaler()):
        if not self.packets[0].errors:
            raise PacktesNotInitiated
        else:
            list_of_arrays = []
            row = []
            columns = []
            for (name, parameter) in zip(self.packets[0].names,
                                         self.packets[0].parameters):
                for (key, list_of_value) in parameter.items():
                    for i in range(len(list_of_value)):
                        columns.append((name,  str(key) + "_" + str(i)))
            for packet in self.packets:
                row = []
                for (name, parameter) in zip(packet.names,
                                             packet.parameters):
                    for (key, list_of_value) in parameter.items():
                        for value in list_of_value:
                            row.append(value)
                list_of_arrays.append(row)

            df = pd.DataFrame(data=list_of_arrays)
            df.columns = pd.MultiIndex.from_tuples(columns)
            df.columns.name = ['Models', 'Parameters']
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
                               normalizer=MinMaxScaler(),
                               only_best=True):
        list_of_signals = []
        list_of_models = []
        list_of_names = []
        list_of_errors = []
        list_of_types = []
        for packet in self.packets:
            if normalized:
                list_of_signals.append(MinMaxScaler().fit_transform(
                                       packet.input_signal.reshape(-1, 1)))
            else:
                list_of_signals.append(packet.input_signal)
            new_set_of_model = []
            if only_best:
                final_error = 100000
                for (errors, model, name) in zip(packet.errors, packet.models,
                                                 packet.names):
                    current_error = sum(errors)
                    if current_error < final_error:
                        best_model = model
                        final_name = name
                        final_error = current_error
                list_of_errors.append(final_error)
                list_of_models.append([item for sublist in best_model
                                       for item in sublist])
                list_of_names.append(final_name)
            else:
                for current_model in packet.models:
                    new_set_of_model.append([item for sublist in current_model
                                             for item in sublist])
                list_of_models.append(new_set_of_model)
                list_of_names.append([x for x in packet.names])
                list_of_errors.append([sum(x) for x in packet.errors])
            list_of_types.append(packet.is_valey)
        return (list_of_signals, list_of_models,
                list_of_names, list_of_errors, list_of_types)

    @classmethod
    def build_from_packets(cls, list_of_packets):
        list_of_peaks = []
        cumul = 0
        for (count, packet) in enumerate(list_of_packets):
            if count == 0:
                list_of_signal = np.array(packet.input_signal)
            else:
                list_of_signal = np.append(list_of_signal, packet.input_signal)
            list_of_peaks.append(cumul + packet.peak_signal)
            cumul += len(packet.input_signal)

        my_cls = cls(list_of_signal, list_of_peaks)
        my_cls.packets = list_of_packets
        return my_cls


class ECG(DefaultSignal):
    def __init__(self, input_signal, indicators, labels,
                 ts=None, filtered=None,
                 heart_rate_ts=None, heart_hate=None,
                 templates_ts=None, templates=None,
                 is_filtered=False, packets=None):

        if len(indicators) != len(labels):
            raise MissMatchError("The indicators and labels must be the same"
                                 + "lenght")

        if indicators is None:
            raise ValueError('You must pass the R peaks.')

        if is_filtered:
            super().__init__(input_signal, indicators, packets)
            self.filtered = input_signal
        else:
            super().__init__(input_signal, indicators, packets)
            self.filtered = None

        self.ts = ts
        self.heart_rate_ts = heart_rate_ts
        self.heart_hate = heart_hate
        self.templates_ts = templates_ts
        self.templates = templates
        self.is_filtered = is_filtered
        self.labels = labels

    def build_packets(self, bandwidth, container=LabeledPacket):
        if self.indicators[0] - bandwidth <= 0:
            raise ValueError('The bandwith must be'
                             + 'less than the first indicator')
        packets = []
        for (ind, label) in zip(self.indicators, self.labels):
            packets.append(container(self.input_signal[(
                           ind-bandwidth):(ind+bandwidth)],
                           bandwidth, label))
        self.packets = packets
        return self.packets

    @classmethod
    def build_from_packets(cls, list_of_packets):
        list_of_peaks = []
        list_of_labels = []
        cumul = 0
        for (count, packet) in enumerate(list_of_packets.copy()):
            if count == 0:
                list_of_signal = np.array(packet.input_signal)
            else:
                list_of_signal = np.append(list_of_signal, packet.input_signal)
            list_of_peaks.append(cumul + packet.peak_signal)
            list_of_labels.append(packet.label)
            cumul += len(packet.input_signal)

        my_cls = cls(list_of_signal, list_of_peaks, list_of_labels)
        my_cls.packets = list_of_packets
        return my_cls

    def args(self):
        if self.packets:
            packets_container = [x.serialize() for x in self.packets]
        else:
            packets_container = None
        return [self.input_signal,
                self.indicators,
                self.labels,
                self.ts,
                self.filtered,
                self.heart_rate_ts,
                self.heart_hate,
                self.templates_ts,
                self.templates,
                self.is_filtered,
                packets_container]


class MathematicalModel():
    def __init__(self,
                 left_function,
                 rigth_function,
                 parameters=None):
        if not parameters:
            parameters = dict()
            parameters['values'] = list(zip(np.linspace(1, 10, 100),
                                        np.linspace(1, 10, 100)))
            parameters['lenght'] = list(zip(np.linspace(35, 350, 10),
                                        np.linspace(35, 350, 10)))
            parameters['cut'] = list(zip(np.linspace(0, 80, 10),
                                     np.linspace(0, 80, 10)))

        self.left_function = left_function
        self.rigth_function = rigth_function
        self.parameters = parameters

    def normalize_amplitude(self, orig_signal, ref_signal):
        xmax = np.max(orig_signal)
        xmin = np.min(orig_signal)
        ymax = np.max(ref_signal)
        ymin = np.min(ref_signal)
        if xmax == xmin:
            return orig_signal
        else:
            return ((ymax-ymin)/(xmax-xmin))*(orig_signal-xmin)+ymin

    def call(self, *, input_signal, values, frequency, percet_ceif,
             peak_signal=None, evalue_model=False, is_valey=False,
             normalize=True):

        if is_valey:
            input_signal = np.multiply(input_signal, -1)

        input_signal = input_signal - np.mean(input_signal)

        # input variables
        left_value, right_value = values
        left_length, rigth_length = frequency
        left_ceif, right_ceif = percet_ceif
        # get the left and right limit's, based on the peak of the input_signal
        size = len(input_signal)

        if not peak_signal:
            peak_signal = self.find_peak(input_signal)

        # Buid the model
        (model, peak_model) = self.build_model(values, frequency)

        # Cut part of the signal
        left_skip = int(np.floor(peak_model * (float(left_ceif) / 100)))
        if left_skip > 0:
            model[:left_skip] = model[:left_skip]
        right_skip = int(np.floor((len(model)-peak_model) * (
                   float(right_ceif) / 100)))
        if right_skip > 0:
            model[:-right_skip] = model[:-right_skip]
        # Extend the model to guarantee that the model its
        # bigguer than input_signal
        left_extend = np.repeat(model[0], size)
        rigth_extend = np.repeat(model[-1], size)
        # Concatenate the extend part of the model
        model = np.concatenate((left_extend, model, rigth_extend), axis=0)
        # afeter this steep peak_model = peak_model + size of extend
        peak_model += size

        model = model[(peak_model - peak_signal):(
                       peak_model + size - peak_signal)]
        peak_model = peak_signal
        model = self.normalize_data(model, input_signal, peak_signal)
        peak_model = peak_signal
        # recalculate the new peak of the model
        if evalue_model:
            (left_error, right_error) = self.metric(input_signal,
                                                    model,
                                                    peak_signal)
            return (model, peak_model, [left_error, right_error])
        else:
            return model, peak_model

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

        left_error = (input_signal[:peak_signal] - model[:peak_signal])
        left_error = [(a**2)*b for a, b in zip(left_error,
                      np.linspace(0.5, 1, peak_signal))]
        left_error = np.sqrt(sum(left_error))

        right_error = (input_signal[peak_signal:] - model[peak_signal:])
        right_error = [(a**2)*b for a, b in zip(right_error,
                       np.linspace(1, 0.5, len(model[peak_signal:])))]
        right_error = np.sqrt(sum(right_error))
        return (left_error, right_error)

    def find_peak(self, input_signal):
        idx_max = np.argmax(input_signal)
        return idx_max

    def find_peak_model(self, input_signal):
        return int(np.round(len(input_signal)/2))

    def normalize_data(self, model, input_signal, peak):

        model[:peak] = self.normalize_amplitude(model[:peak],
                                                input_signal[:peak])

        model[peak:] = self.normalize_amplitude(model[peak:],
                                                input_signal[peak:])

        return model

    def build_model(self, values, frequency):
        left_value, right_value = values
        left_length, rigth_length = frequency

        r_left = int(left_length/2)
        r_rigth = int(rigth_length/2)
        left_model = self.left_function(left_length, left_value)[:r_left]
        rigth_model = self.rigth_function(rigth_length, right_value)[r_rigth:]
        model = np.concatenate((left_model, rigth_model), axis=0)
        return model, r_left


class MexicanHat(MathematicalModel):
    def __init__(self, parameters=None):
        super().__init__(sci_signal.ricker, sci_signal.ricker, parameters)
        self.name = "mexican_hat"

    def mexican_hat_func(self, n, sigma):
        x = np.linspace(-40, 40, n)
        C = (2/np.sqrt(3*sigma)) * (np.pi**1/4)
        return C*(1 - x**2/(sigma**2)) * np.exp(-x**2/(2*sigma**2))


class Gaussian(MathematicalModel):
    def __init__(self, parameters=None):
        if not parameters:
            parameters = dict()
            parameters['values'] = list(zip(np.linspace(0.01, 2, 100),
                                        np.linspace(0.01, 2, 100)))
            parameters['lenght'] = list(zip(np.linspace(35, 350, 10),
                                        np.linspace(35, 350, 10)))
            parameters['cut'] = list(zip(np.linspace(0, 80, 10),
                                     np.linspace(0, 80, 10)))
        super().__init__(self.gaussian_func, self.gaussian_func, parameters)
        self.name = "gaussian"

    def gaussian_func(self, n, sigma):
        x = np.linspace(-2, 2, n)
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp((-(x-0)**2)/2*sigma**2)


class Rayleigh(MathematicalModel):
    def __init__(self, parameters=None):
        super().__init__(self.rayleigh_func, self.rayleigh_func, parameters)
        self.name = "rayleigh"

    def rayleigh_func(self, n, scale):
        x = np.linspace(0, 2, n)
        return (x/(scale**2)) * np.e**((-x**2)/2*scale**2)

    def find_peak_model(self, input_signal):
        idx_max = np.argmax(input_signal)
        return idx_max

    def build_model(self, values, frequency):
        left_value, right_value = values
        left_length, rigth_length = frequency
        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)
        model = np.concatenate((left_model[::-1], rigth_model[1:]), axis=0)
        return model, np.argmax(model)


class LeftInverseRayleigh(Rayleigh):
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.name = "left_inverse_rayleigh"

    def build_model(self, values, frequency):
        left_value, right_value = values
        left_length, rigth_length = frequency
        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)
        rigth_model = self.rigth_function(rigth_length, right_value)
        left_model = self.normalize_amplitude(left_model, rigth_model)
        model = np.concatenate((left_model[::-1]*-1, rigth_model[1:]), axis=0)
        return model, np.argmax(model)


class RightInverseRayleigh(Rayleigh):
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.name = "right_inverse_rayleigh"

    def build_model(self, values, frequency):
        left_value, right_value = values
        left_length, rigth_length = frequency
        left_model = self.left_function(left_length, left_value)
        rigth_model = self.rigth_function(rigth_length, right_value)
        left_model = self.normalize_amplitude(left_model, rigth_model)
        model = np.concatenate((left_model[::-1], rigth_model[1:]*-1), axis=0)
        return model, np.argmax(model)
