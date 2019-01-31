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
    list_ecg = my_ecg.build_packets(25, fb.DefaultPacket)

    norm_parameter = dict()
    norm_parameter['values'] = list(zip(np.linspace(3, 13, 10),
                                    np.linspace(3, 13, 10)))
    norm_parameter['lenght'] = list(zip(np.linspace(15, 20, 5),
                                    np.linspace(15, 20, 5)))
    parameter = dict()
    parameter['values'] = list(zip(np.linspace(3, 13, 10),
                               np.linspace(3, 13, 10)))
    parameter['lenght'] = list(zip(np.linspace(15, 20, 5),
                               np.linspace(15, 20, 5)))

    template_gaussian = fb.TemplateEvaluetor(fb.Gaussian(), norm_parameter)
    template_mexican = fb.TemplateEvaluetor(fb.MexicanHat(), parameter)
    template_rayleigh = fb.TemplateEvaluetor(fb.Rayleigh(), parameter)
    template_left_rayleigh = fb.TemplateEvaluetor(fb.LeftInverseRayleigh(),
                                                  parameter)
    template_right_rayleigh = fb.TemplateEvaluetor(fb.RightInverseRayleigh(),
                                                   parameter)

    list_of_templates = [template_gaussian, template_mexican,
                         template_rayleigh, template_left_rayleigh,
                         template_right_rayleigh]

    lst_of_packets = utils.mp_signal_apply(utils.consume_packet_helper,
                                   list_ecg, list_of_templates)

    my_ecg.packets = lst_of_packets
    return my_ecg


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    results = main()
    error = results.getErrorsTable()
