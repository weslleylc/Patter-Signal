# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:51:11 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
from scipy import signal as sci_signal
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import peakutils
import scipy.stats
from matmodel import WTdelineator as wav
from pywt import Wavelet, wavedec
import operator
import json

registry = {}


class Error(Exception):
    """Base class for other exceptions"""
    pass


class PacketsNotInitiated(Error):
    """Raised when the pakctes are not initiated."""
    pass


class MissMatchError(Error):
    """Raised when the pakctes are not initiated."""
    pass


def register_class(target_class):
    """Add the name class in registry dictionary."""
    registry[target_class.__name__] = target_class


def deserialize(data):
    """Call an class inside registry dictionary with data as input."""
    params = json.loads(data)
    name = params['class']
    target_class = registry[name]
    return target_class(*params['args'])


class Meta(type):
    """Metaclass that's add serialization for all classes."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls


class NumpyEncoder(json.JSONEncoder):
    """Json enconder for numpy array."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BetterSerializable(object):
    """Base class for json serialization."""
    def __init__(self, *args):
        self.args = args

    def serialize(self):
        """Just call's dumps from json with name and data class."""
        return json.dumps({
                'class': self.__class__.__name__,
                'args': self.input_args(),
                },
                cls=NumpyEncoder)

    def input_args(self):
        raise NotImplementedError("You must to implement args.")


class RegisteredSerializable(BetterSerializable, metaclass=Meta):
    """Inherits from BetterSerializable and Meta. Base for serialization."""
    pass


class DefaultPacket(RegisteredSerializable):
    """Deftauls class for packet container's.

    This class help's to encapsulate the segmentation in a signal (a hearbeat
    in ECG, for example). Contains all informations about the metrics utilized
    in a matemathical model

    Attributes:
        input_signal: np.array for a segmentation of a whole signal.
        peak_signal: A int representing the idex of bigger peak.
        models: A lista of the final models for each matemathical model
        evalueted.
        errors. A lista of the final errors for each evaluetion of an
        matematical model.
        parameters = the final parameters for the best's evaluetion of all
        mathemathical model.
        names: A list with all names for each matematical model utilized
        is_valey: A boolean that's inform if the bigguer peak is a valey or
        note.
        remove_trends: if True remove the mean of the signal
    """
    def __init__(self, input_signal, peak_signal=None, models=None,
                 errors=None, parameters=None, names=None, is_valey=None,
                 remove_trends=True):
        """Initialize a packet and asegurate that input signal and models,
        contains only numpy arrays"""
        input_signal = np.array(input_signal)
        if remove_trends:
            m = np.mean(input_signal[[0, -1]])
            input_signal = np.subtract(input_signal, m)

        self.input_signal = input_signal
        self.peak_signal = peak_signal
        if type(models) == list:
            for i, m in enumerate(models):
                if type(m) == list:
                    for j, mm in enumerate(m):
                        models[i][j] = np.array(mm)
                else:
                    m = np.array(m)
        self.models = models
        self.parameters = parameters
        self.errors = errors
        self.names = names
        if not is_valey:
            is_valey = self.direction_peak(input_signal)
        self.is_valey = is_valey

    def direction_peak(self, input_signal, factor=0.5):
        """return is the peak is or not a valey.

        perform's a valey decetion of an input_signal

        Args:
            input_signal : array [n_samples]
                A array that represents a signal
            factor : int
                the max percentage diferency to considerate a positive peak
                more important than a negative peak.
        Returns:
            (left_error, right_error): tuple(float,float)
                A tuple that contains the calculated left and right
                mean squared error
        Raises:
            IndexError: An error occurred accessing input_signal.
        """
        id_max = np.argmax(input_signal)
        id_min = np.argmin(input_signal)
        if (input_signal[id_max] < np.abs(input_signal[id_min]) * factor):
            return True
        return False

    def input_args(self):
        """return a list with all attributes."""
        return [self.input_signal,
                self.peak_signal,
                self.models,
                self.errors,
                self.parameters,
                self.names,
                self.is_valey]


class LabeledPacket(DefaultPacket):
    def __init__(self, input_signal, peak_signal=None, label=None,
                 models=None, errors=None, parameters=None,
                 names=None, is_valey=False, remove_trends=True):
        """Initialize a packet and call's DefaultPacket"""
        super().__init__(input_signal, peak_signal, models, errors,
                         parameters, names, is_valey, remove_trends)
        self.label = label

    def input_args(self):
        """return a list with all attributes."""
        return [self.input_signal,
                self.peak_signal,
                self.label,
                self.models,
                self.errors,
                self.parameters,
                self.names,
                self.is_valey]


class DefaultSignal(RegisteredSerializable):
    """Deftaults class for signal's container's.

    This class help's to encapsulate an whole signal (an ECG, for example).
    Contains methods that return tables with all informations contained on
    the packets.

    Attributes:
        input_signal: np.array for a whole signal.
        indicators: list that's contains all peaks for each packet
        packets: A lista of the the packets
    """
    def __init__(self, input_signal, indicators=None, packets=None):
        """Initiate the signal. To utilized pre existent packets see
        build_from_packets. Dont pass packets as input, they stay here
        only for deseriliazable purpose. """
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
        """Return a list of packets.

        Build the packets using the input_signal and indicators.

        Args:
            bandwidth : A int number that represent the half lenght of
            segmented signal.
            container : A class to encapsulate the packet.
            normalizer : A boolean that indicate if the segmented singal
            must be normalized or not.
            normalizer: normalizer object, that will be used to aplly
            normalization on the segmented signal.
        Returns:
            packets: A list of packets
        Raises:
            IndexError: An error occurred accessing input_signal.
        """
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
        """Return a dataframe with the errors for evalueted models

        Return a dataframe wich each row represent's the errors for each
        matemathical model for a segmented signal. Optionally  for each
        piecewise of an model, normalize separatly all erros reference to this
        part of the model.
        Args:
            normalizer : A boolean that indicate if the errors
            must be normalized or not.
            normalizer: normalizer object, that will be used to aplly
            normalization on the errors.
        Returns:
            df: dataframe for all errors named by columns
        Raises:
            PacketsNotInitiated: An error occurred if the packets are not
            initilazed before.
        """
        if not self.packets[0].errors:
            raise PacketsNotInitiated
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
                    # normalize only between each part of the model, for each
                    # distribution
                    for j in df.columns.levels[1].values:
                        df.iloc[i].loc[:, j] = normalizer.fit_transform(
                            df.iloc[i].loc[:, j].values.reshape(-1, 1)
                            ).reshape(1, -1)[0]

            return df

    def get_param_table(self, normalized=True, normalizer=MinMaxScaler()):
        """Return a dataframe with the parans for evalueted models

        Return a dataframe wich each row represent's the parameters for each
        matemathical model for a segmented signal. Optionally  for each
        piecewise of an model, normalize separatly all parameters reference to
        this part of the model.
        Args:
            normalizer : A boolean that indicate if the errors
            must be normalized or not.
            normalizer: normalizer object, that will be used to aplly
            normalization on the parameters.
        Returns:
            df: dataframe for all parameters named by columns
        Raises:
            PacketsNotInitiated: An error occurred if the packets are not
            initilazed before.
        """
        if not self.packets[0].errors:
            raise PacketsNotInitiated
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
        """Return a list with all segmented signal"""
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
        """return a tuple with five list that contais all informations about
        models and signals.

        For each list on the tuple has cople of elemets like signals, models,
        or the names of models. Optionally normalize the signals. By the way,
        Optionally return only the best mathematical model
        Args:
            normalizer : A boolean that indicate if the signal
            must be normalized or not.
            normalizer: Normalizer object, that will be used to aplly
            normalization on the signal.
            only_best: A boolean that indicate to return all models or only
            the best model
        Returns:
            list_of_signals: list of segmented signals
            list_of_models: list of numpy array picewise evalueted models
            list_of_names: lisft of names od the models
            list_of_errors: list of list of errors
            list_of_types: listy of types of peaks (valey or not)
        Raises:
            PacketsNotInitiated: An error occurred if the packets are not
            initilazed before.
            TypeError : an error ocorred if the models aren't list of numpy
            arrays
        """
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
        """contruct a signal using a pre existent lisft of packets

        Concatenate all segmented inputs to build a whole signal
        Args:
            list_of_packets : A list contains packets
        Returns:
            my_cls: a object of this class
        Raises:

            ValueError : an error ocorred if the packets have missing values.
        """
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
    """Deftaults class for ECG signal's container's.

    This class help's to encapsulate an whole ECG signal.
    Contains methods that return tables with all informations contained on
    the packets.

    Attributes:
        input_signal: np.array for a whole signal.
        indicators: list that's contains all peaks for each packet
        labels:
        ts:
        filtered:
        heart_rate_ts:
        packets: A lista of the the packets
    """
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

        input_signal = np.array(input_signal)
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

    def input_args(self):
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


class MathematicalModel(object):
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
        return model, int(left_length)


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


class RandomExtraction(RegisteredSerializable):
    def __init__(self, n_classifiers=15, offset=50, n_features=50,
                 results=None, random_matrix=None):
        self.n_classifiers = n_classifiers
        self.n_features = n_features
        self.offset = offset
        if not results:
            self.results = []
        else:
            self.results = np.array(results)
        if not random_matrix:
            self.random_matrix = np.random.randn(2 * self.offset *
                                                 self.n_classifiers,
                                                 self.n_features)
        else:
            self.random_matrix = np.array(random_matrix)

    def evaluete_signal(self, input_signal, label=''):

        if len(input_signal) != 2 * self.offset:
            input_signal = signal.resample(input_signal, 2 * self.offset)
        results = []
        for i in range(self.n_classifiers):
            x = np.matmul(input_signal, self.random_matrix[(0 + (2 *
                          self.offset * (i))):(2 * self.offset * (1 + i)), :])
            results = np.concatenate((results, x))

        self.results.append(np.concatenate((results, [label])))

    def get_table_erros(self):
        return pd.DataFrame(data=self.results)

    def input_args(self):
        return [self.n_classifiers,
                self.offset,
                self.n_features,
                self.results,
                self.random_matrix]


class MorfologyDetector(RegisteredSerializable):
    def __init__(self, a=[0, 40],b=[75, 85],c=[95,105] ,d=[150,180],results=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if not results:
            self.results = []
        else:
            self.results = np.array(results)

    # Function adapted from https://github.com/mondejar/ecg-classification
    def evaluete_signal(self, input_signal, peak_signal, is_valey=False, label=''):
        if len(input_signal) != 180:
            input_signal = signal.resample(input_signal, 180)
        results = []
        R_pos = peak_signal

        R_value = input_signal[R_pos]
        my_morph = np.zeros((4))
        y_values = np.zeros(4)
        x_values = np.zeros(4)
        # Obtain (max/min) values and index from the intervals
        [x_values[0], y_values[0]] = max(enumerate(input_signal[self.a[0]:self.a[1]]), key=operator.itemgetter(1))
        [x_values[1], y_values[1]] = min(enumerate(input_signal[self.b[0]:self.b[1]]), key=operator.itemgetter(1))
        [x_values[2], y_values[2]] = min(enumerate(input_signal[self.c[0]:self.c[1]]), key=operator.itemgetter(1))
        [x_values[3], y_values[3]] = max(enumerate(input_signal[self.d[0]:self.d[1]]), key=operator.itemgetter(1))

        x_values[1] = x_values[1] + self.a[0]
        x_values[2] = x_values[2] + self.b[0]
        x_values[3] = x_values[3] + self.d[0]

        # Norm data before compute distance
        x_max = max(x_values)
        y_max = max(np.append(y_values, R_value))
        x_min = min(x_values)
        y_min = min(np.append(y_values, R_value))

        R_pos = (R_pos - x_min) / (x_max - x_min)
        R_value = (R_value - y_min) / (y_max - y_min)

        for n in range(0, 4):
            x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
            y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
            x_diff = (R_pos - x_values[n])
            y_diff = R_value - y_values[n]
            my_morph[n] = np.linalg.norm([x_diff, y_diff])
        self.results.append(np.concatenate((my_morph, [label])))
        return my_morph

    def get_table_erros(self):
        columns = ["md{}".format(i) for i in range(4)]
        columns.extend(['label'])
        return pd.DataFrame(data=self.results, columns=columns)

    def input_args(self):
        return [self.a, self.b, self.c, self.d, self.results]


class HOSExtraction(RegisteredSerializable):
    def __init__(self, n_intervals=6, results=None):
        self.n_intervals = n_intervals
        if not results:
            self.results = []
        else:
            self.results = results

    # Compute the HOS descriptor for a heartbeat
    # Skewness (3 cumulant) and kurtosis (4 cumulant)
    # Function adapted from https://github.com/mondejar/ecg-classification
    def evaluete_signal(self, input_signal, label=''):
        lag = int(round(len(input_signal) / self.n_intervals))
        hos_b = np.zeros(((self.n_intervals - 1) * 2))
        for i in range(0, self.n_intervals - 1):
            pose = (lag * (i + 1))
            interval = input_signal[int(pose - (lag / 2)):int(pose + (lag / 2))]

            # Skewness
            hos_b[i] = scipy.stats.skew(interval, 0, True)

            if np.isnan(hos_b[i]):
                hos_b[i] = 0.0

            # Kurtosis
            hos_b[(self.n_intervals - 1) + i] = scipy.stats.kurtosis(interval, 0, False, True)
            if np.isnan(hos_b[(self.n_intervals - 1) + i]):
                hos_b[(self.n_intervals - 1) + i] = 0.0
        self.results.append(np.concatenate((hos_b, [label])))
        return hos_b

    def get_table_erros(self):
        skewness = ["s{}".format(i) for i in range(self.n_intervals -1)]
        kurtosis = ["k{}".format(i) for i in range(self.n_intervals -1)]
        kurtosis.extend(['label'])
        return pd.DataFrame(data=self.results, columns=skewness.extend(kurtosis))

    def input_args(self):
        return [self.n_intervals,
                self.results]

class HermiteExtraction(RegisteredSerializable):
    def __init__(self, results=None):
        if not results:
            self.results = []
        else:
            self.results = results

    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.hermite.html
    # Support Vector Machine-Based Expert System for Reliable Heartbeat Recognition
    # 15 hermite coefficients!
    def evaluete_signal(self, input_signal, label=''):
        coeffs_hbf = np.zeros(15, dtype=float)
        coeffs_HBF_3 = np.polynomial.hermite.hermfit(range(0, len(input_signal)), input_signal, 3)  # 3, 4, 5, 6?
        coeffs_HBF_4 = np.polynomial.hermite.hermfit(range(0, len(input_signal)), input_signal, 4)
        coeffs_HBF_5 = np.polynomial.hermite.hermfit(range(0, len(input_signal)), input_signal, 5)

        coeffs_hbf = np.concatenate((coeffs_HBF_3, coeffs_HBF_4, coeffs_HBF_5))
        self.results.append(np.concatenate((coeffs_hbf, [label])))
        return coeffs_hbf

    def get_table_erros(self):
        hermite_3 = ["hbf_3{}".format(i) for i in range(5)]
        hermite_4 = ["hbf_4{}".format(i) for i in range(5)]
        hermite_5 = ["hbf_5{}".format(i) for i in range(5)]
        columns = []
        columns.extend(hermite_3)
        columns.extend(hermite_4)
        columns.extend(hermite_5)
        columns.extend(['label'])

        return pd.DataFrame(data=self.results,
                            columns=columns)

    def input_args(self):
        return [self.results]


class ULBP1DExtraction(RegisteredSerializable):
    def __init__(self, results=None):
        self.uniform_pattern_list = np.array(
            [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126,
             127, 128,
             129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247,
             248,
             249, 251, 252, 253, 254, 255])
        if not results:
            self.results = []
        else:
            self.results = results

    ## Function adapted from https://github.com/mondejar/ecg-classification

    # Compute the uniform LBP 1D from input_signal with neigh equal to number of neighbours
    # and return the 59 histogram:
    # 0-57: uniform patterns
    # 58: the non uniform pattern
    # NOTE: this method only works with neigh = 8
    def compute_Uniform_LBP(self, input_signal, neigh=8):
        hist_u_lbp = np.zeros(59, dtype=float)
        aux = int(neigh / 2)

        for i in range(aux, len(input_signal) - aux):
            pattern = np.zeros(neigh)
            ind = 0
            for n in list(range(-aux, 0)) + list(range(1, aux + 1)):
                if input_signal[i] > input_signal[i + n]:
                    pattern[ind] = 1
                ind += 1
            # Convert pattern to id-int 0-255 (for neigh == 8)
            pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

            # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
            if pattern_id in self.uniform_pattern_list:
                pattern_uniform_id = int(np.argwhere(self.uniform_pattern_list == pattern_id))
            else:
                pattern_uniform_id = 58  # Non uniforms patternsuse

            hist_u_lbp[pattern_uniform_id] += 1.0

        return hist_u_lbp

    def evaluete_signal(self, input_signal, label=''):
        hist_u_lbp = self.compute_Uniform_LBP(input_signal)
        self.results.append(np.concatenate((hist_u_lbp, [label])))

    def get_table_erros(self):
        results = np.array(self.results)
        size = results.shape[1] -1
        hist = ["h{}".format(i) for i in range(size)]
        hist.extend(['label'])
        return pd.DataFrame(data=self.results, columns=hist)

    def input_args(self):
        return [self.results]


class LBP1DExtraction(RegisteredSerializable):
    def __init__(self, results=None):
        self.uniform_pattern_list = np.array(
            [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126,
             127, 128,
             129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247,
             248,
             249, 251, 252, 253, 254, 255])
        if not results:
            self.results = []
        else:
            self.results = results

    ## Function adapted from https://github.com/mondejar/ecg-classification

    def compute_LBP(self, input_signal, neigh=4):
        hist_u_lbp = np.zeros(np.power(2, neigh), dtype=float)

        aux = int(neigh / 2)
        for i in range(aux, len(input_signal) - aux):
            pattern = np.zeros(neigh)
            ind = 0
            for n in list(range(-aux, 0)) + list(range(1, aux + 1)):
                if input_signal[i] > input_signal[i + n]:
                    pattern[ind] = 1
                ind += 1
            # Convert pattern to id-int 0-255 (for neigh == 8)
            pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

            hist_u_lbp[pattern_id] += 1.0

        return hist_u_lbp

    def evaluete_signal(self, input_signal, label=''):
        hist_u_lbp = self.compute_LBP(input_signal)
        self.results.append(np.concatenate((hist_u_lbp, [label])))

    def get_table_erros(self):
        results = np.array(self.results)
        size = results.shape[1] -1
        hist = ["h{}".format(i) for i in range(size)]
        hist.extend(['label'])
        return pd.DataFrame(data=self.results, columns=hist)

    def input_args(self):
        return [self.results]


class WaveletTransform(RegisteredSerializable):
    def __init__(self, results=None):

        if not results:
            self.results = []
        else:
            self.results = results

    # Function adapted from https://github.com/mondejar/ecg-classification
    def evaluete_signal(self, input_signal, label=''):
        db1 = Wavelet('db1')
        coeffs = wavedec(input_signal, db1, level=3)
        wavel = coeffs[0]
        self.results.append(np.concatenate((wavel, [label])))

    def get_table_erros(self):
        results = np.array(self.results)
        size = results.shape[1] -1
        columns = ["w{}".format(i) for i in range(size)]
        columns.extend(['label'])
        return pd.DataFrame(data=results, columns=columns)

    def input_args(self):
        return [self.results]


class WaveletExtraction(RegisteredSerializable):
    def __init__(self, epi_qrs=[0.5, 0.5, 0.5, 0.5, 0.5],
                 epi_p=0.02, epi_q=0.25, results=None):
        self.epi_qrs = epi_qrs
        self.epi_p = epi_p
        self.epi_q = epi_q
        if not results:
            self.results = []
        else:
            self.results = results

    def evaluete_signal(self, input_signal, fs=360, filtered=False, label=''):
        # Number of samples in the signal
        N = input_signal.shape[0]
        # Create the filters to apply the algorithme-a-trous
        Q = wav.waveletFilters(N, fs)
        # Perform signal decomposition
        if filtered:
            w = wav.waveletDecomp(self.smooth(input_signal, 3), Q)
        else:
            w = wav.waveletDecomp(input_signal, Q)
        results = []
        for i, cd in enumerate(w):
            [positive, negative] = self.count_peaks(cd,
                                                    rms(cd) * self.epi_qrs[i])
            results = np.concatenate((results, [positive, negative]))
        self.results.append(np.concatenate((results, [label])))

    def get_table_erros(self):
        return pd.DataFrame(data=self.results, columns=['w1p', 'w1n',
                                                        'w2p', 'w2n',
                                                        'w3p', 'w3n',
                                                        'w4p', 'w4n',
                                                        'w5p', 'w5n',
                                                        'label'])

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

    def input_args(self):
        return [self.epi_qrs,
                self.epi_p,
                self.epi_q,
                self.results]

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth


def rms(x):
    return np.sqrt(np.mean(x**2))
