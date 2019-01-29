# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:47:36 2019

@author: Weslley L. Caldas
"""
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from scipy.misc import electrocardiogram
from matmodel import utils, feature_builder as fb


def main():
    ecg_signal = electrocardiogram()
    ecg_signal = ecg_signal - np.mean(ecg_signal)

    (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate
     ) = ecg.ecg(signal=ecg_signal, sampling_rate=360, show=False)

    my_ecg = fb.ECG(input_signal=filtered, indicators=rpeaks,
                    is_filtered=False)
    list_ecg = my_ecg.buildPackets(25)

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
                                   list_ecg[:10], list_of_templates)
    names = ['left_error', 'left_value', 'left_length',
             'right_error', 'right_value', 'right_length']

    columns = []
    for t in list_of_templates:
        current_name = t.mat_model.__class__.__name__
        columns.extend([x + current_name for x in names])

    df = pd.DataFrame(result, columns=columns)
    return df


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    results = main()
