# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:03:33 2019

@author: Weslley L. Caldas
"""

import numpy as np
import pandas as pd
from biosppy.signals import ecg
from scipy.misc import electrocardiogram
import mat4py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matmodel import utils, feature_builder as fb
import pickle


def main():
    dataset = np.load('./data/morfology.npy').item()
    n = 600
    list_ecg = dataset['signal'][:n]
    list_peaks = dataset['peaks'][:n]
    list_labels = dataset['label'][:n]

    norm_parameter = list(zip(np.linspace(3, 13, 10), np.linspace(3, 13, 10)))
    norm_lenght = list(zip(np.linspace(15, 20, 5), np.linspace(15, 20, 5)))
    parameter = list(zip(np.linspace(3, 13, 10), np.linspace(3, 13, 10)))
    lenght = list(zip(np.linspace(15, 20, 5), np.linspace(15, 20, 5)))

    template_gaussian = fb.TemplateEvaluetor(fb.Gaussian(),
                                             norm_parameter, norm_lenght)
    template_mexican = fb.TemplateEvaluetor(fb.MexicanHat(), parameter, lenght)
    template_rayleigh = fb.TemplateEvaluetor(fb.Rayleigh(), parameter, lenght)
    template_left_rayleigh = fb.TemplateEvaluetor(fb.LeftInverseRayleigh(),
                                                  parameter, lenght)
    template_right_rayleigh = fb.TemplateEvaluetor(fb.RightInverseRayleigh(),
                                                   parameter, lenght)

    list_of_templates = [template_gaussian, template_mexican,
                         template_rayleigh, template_left_rayleigh,
                         template_right_rayleigh]

    result = utils.evaluete_models(utils.consume_process,
                                   list_ecg, list_of_templates,list_peaks)
    names = ['left_error', 'left_value', 'left_length',
             'right_error', 'right_value', 'right_length']

    columns = []
    for t in list_of_templates:
        current_name = t.mat_model.__class__.__name__
        columns.extend([x + current_name for x in names])

    df = pd.DataFrame(result, columns=columns)
    df['labels'] = list_labels
    return df


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    results = main()

#
#
#dataset = mat4py.loadmat('./data/morfology.mat')
#for i in range(600):
#    dataset['signal'][i] = [item for sublist in dataset['signal'][i] for item in sublist]
#     
#np.save('morfology.npy', dataset)
#dataset = np.load('./data/morfology.npy').item()



