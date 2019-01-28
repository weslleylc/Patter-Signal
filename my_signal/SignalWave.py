# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:51:11 2019

@author: Weslley L. Caldas
"""

from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from scipy.misc import electrocardiogram
import matplotlib.pyplot as plt
from scipy.stats import rayleigh


# class wave:
#     self._indicators = []
#     self._wave = []
#     def __init__(self, value):
#         self.value = value

# class ECG(wave):
#     def __init__(self):

# class EEG(wave):
#     def __init__(self, value):
#         super(__class__, self).__init__(value)

# class Packet():
#     self_patterns = []
#     self_patterns = []
#     def __init__(self):
        
        


    # def rayleight(self, lenght, value):
    #     if (value <= 0) or (value >= 1):
    #         raise ValueError('The value must be between (0,1).')
    #     mean, var, skew, kurt = rayleigh.stats(moments='mvsk')
    #     x = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(value), lenght)
    #     return rayleigh.pdf(x)

    # def find_crosspoint(self, model, direction=True):
    #     aux = model[0]
    #     count = 0
    #     if direction:
    #         index = range(0, len(model))
    #     else:
    #         index = range(len(model), 0)

    #     for i in index:
    #         if model[i] < 0.1**10:
    #             return count
    #         else:
    #             count += 1
    #             aux = model[i]
    #     return count



class MathematicalModel():
    def __init__(self,
                 left_function,
                 rigth_function):

        self.left_function = left_function
        self.rigth_function = rigth_function

    def call(self, *, wave, left_value, left_length,
             right_value, rigth_length):

        (left_wave, rigth_wave) = self.split_wave(wave)

        model = self.build_model(left_value, left_length,
                                 right_value, rigth_length)

        if(type(model) != np.array):
            model = np.array(model).reshape(-1, 1)
        if(type(wave) != np.array):
            wave = np.array(wave).reshape(-1, 1)

        size = len(wave)

        left_extend = np.repeat(model[0], size).reshape(-1, 1)
        rigth_extend = np.repeat(model[-1], size).reshape(-1, 1)

        model = np.concatenate((left_extend, model, rigth_extend), axis=0)

        peak_sig = self.find_peak(wave)
        peak_model = self.find_peak_model(model)

        left_limit = peak_sig
        rigth_limit = size - peak_sig

        model = model[(peak_model-left_limit):(peak_model+rigth_limit)]
        normalized = self.normalize_data(model, wave,
                                         peak_sig, left_limit)
        return normalized

    def split_wave(self, wave):
        """
        This function return a tuple with the left and right part of the
        wave, splited on their center.
        """
        idx_max = np.argmax(wave)
        return (wave[:idx_max], wave[idx_max:])

    def metric(self, y_true, y_pred):
        """
        This function return the evaluetion given a siginal and a mathematical
        model
        """
        return mean_squared_error(y_true, y_pred)

    def find_peak(self, wave):
        idx_max = np.argmax(wave)
        return idx_max

    def find_peak_model(self, wave):
        idx_max = np.argmax(wave)
        return idx_max

    def normalize_data(self, model, wave,
                       peak_sig, peak_model):

        normalized_wave = MinMaxScaler().fit_transform(wave)

        left_model = MinMaxScaler(feature_range=(
                                  np.min(normalized_wave[:peak_sig]),
                                  np.max(normalized_wave[:peak_sig]))).fit(
                                  model[:peak_model]).transform(
                                  model[:peak_model])

        right_model = MinMaxScaler(feature_range=(
                                   np.min(normalized_wave[peak_sig:]),
                                   np.max(normalized_wave[peak_sig:]))).fit(
                                   model[peak_model:]).transform(
                                   model[peak_model:])

        normalized_model = np.concatenate((left_model, right_model))
        return (normalized_model, normalized_wave)

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
        super().__init__(signal.ricker, signal.ricker)


class Gaussian(MathematicalModel):
    def __init__(self):
        super().__init__(signal.gaussian, signal.gaussian)


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



ecg = electrocardiogram()
ecg = ecg - np.mean(ecg)
ecg = ecg[100:150]

# m=MexicanHat()
# A=m.call(wave=ecg, left_value=6, left_length=27,
#               right_value=5, rigth_length=23)

r = RightInverseRayleigh()
B = r.call(wave=ecg, left_value=15, left_length=50,
              right_value=10, rigth_length=50)
plt.plot(B[0],'-',B[1],'-')
plt.show()

# a = r.rayleigh_func(100,20)
# # c = r.find_crosspoint(a)
# b = r.rayleigh_func(150,10)
# e = np.concatenate((b[::-1]*-1, a[1:]))
# plt.plot(e)

