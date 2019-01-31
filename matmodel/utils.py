# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:50:18 2019

@author: Weslley L. Caldas
"""
import numpy as np
import multiprocessing as mp


def consume_process(args):
    print("child")
    return consume_evaluetor(args[0], args[1], args[2], args[3], args[4])


def mp_apply(function, list_of_evaluetor, input_signal,
             list_of_peaks=None, workers=2):
    if not list_of_peaks:
        list_of_peaks = [None for x in list_of_evaluetor]
    with mp.Pool(processes=workers) as pool:
        result = pool.map(function, [(input_signal,
                                     list_of_evaluetor[index].mat_model,
                                     list_of_evaluetor[index].parameter,
                                     list_of_evaluetor[index].lenght,
                                     list_of_peaks[index])
                          for index in range(len(list_of_evaluetor))])
    pool.close()
#    return [item for sublist in result for item in sublist]
    return result


def evaluete_models(function, packets_of_signals,
                    list_of_evaluetor, list_of_peaks=None):
    evalueted_signals = []
    for sig in packets_of_signals:
        input_signal = sig.signal
        peak_signal = sig.peak_signal
        result = mp_apply(function, list_of_evaluetor,
                          input_signal, list_of_peaks, 2)
        evalueted_signals.append(result)
    return evalueted_signals


def mp_signal_apply(function, packets_of_signals,
                    list_of_evaluetor, workers=2):

    with mp.Pool(processes=workers) as pool:
        result = pool.map(function, [(packets_of_signals[index],
                                     list_of_evaluetor.copy())
                          for index in range(len(packets_of_signals))])
    pool.close()
    return result


def consume_packet_helper(args):
    print(1)
    return consume_packet(args[0], args[1])


def consume_packet(packet, list_of_evaluetor):
    list_of_error = []
    list_of_parameters = []
    list_of_best_model = []
    list_of_names = []
    for evaluetor in list_of_evaluetor:
        (erros, parameters, best_model) = consume_signal(packet, evaluetor)
        list_of_error.append(erros)
        list_of_parameters.append(parameters)
        list_of_best_model.append(best_model)
        list_of_names.append(evaluetor.mat_model.name)
    packet.errors = list_of_error
    packet.parameters = list_of_parameters
    packet.models = list_of_best_model
    packet.names = list_of_names
    return packet
#    return (list_of_error, list_of_parameters, list_of_best_model)


def consume_signal(packet, evaluetor):

    parameters = dict()
    final_left_error = np.inf
    final_right_error = np.inf

    input_signal = packet.input_signal
    peak_signal = packet.peak_signal
    mat_model = evaluetor.mat_model

    for (left_value, right_value) in evaluetor.parameters['values']:
        for (left_length, rigth_length) in evaluetor.parameters['lenght']:

            (model, error) = mat_model.call(input_signal=input_signal,
                                            left_value=left_value,
                                            left_length=left_length,
                                            right_value=right_value,
                                            rigth_length=rigth_length,
                                            peak_signal=peak_signal)
            left_model = model[0]
            left_error = error[0]
            right_model = model[1]
            right_error = error[1]

            if final_left_error > left_error:
                final_left_model = left_model
                final_left_error = left_error
                final_left_value = left_value
                final_left_length = left_length

            if final_right_error > right_error:
                final_right_model = right_model
                final_right_error = right_error
                final_right_value = right_value
                final_right_length = rigth_length

    best_model = [final_left_model, final_right_model]
    errors = [final_left_error, final_right_error]
    parameters['values'] = [final_left_value, final_right_value]
    parameters['length'] = [final_left_length, final_right_length]

    return (errors, parameters, best_model)


def consume_evaluetor(input_signal, model, parameter,
                      lenght, peak_signal=None):
    final_left_error = np.inf
    final_right_error = np.inf

    for (left_value, right_value) in parameter:
        for (left_length, rigth_length) in lenght:

            (left_model, left_error,
             right_model, right_error) = model.call(input_signal=input_signal,
                                                    left_value=left_value,
                                                    left_length=left_length,
                                                    right_value=right_value,
                                                    rigth_length=rigth_length,
                                                    peak_signal=peak_signal)
            if final_left_error > left_error:
                final_left_model = left_model
                final_left_error = left_error
                final_left_value = left_value
                final_left_length = left_length

            if final_right_error > right_error:
                final_right_model = right_model
                final_right_error = right_error
                final_right_value = right_value
                final_right_length = rigth_length

    results = dict()
    results['final_left_model'] = final_left_model
    results['final_left_error'] = final_left_error
    results['final_left_value'] = final_left_value
    results['final_left_length'] = final_left_length
    results['final_right_model'] = final_right_model
    results['final_right_error'] = final_right_error
    results['final_right_value'] = final_right_value
    results['final_right_length'] = final_right_length

    return results



#
#def consume_evaluetor(input_signal, model, parameter,
#                      lenght, peak_signal=None):
#    final_left_error = np.inf
#    final_right_error = np.inf
#
#    for (left_value, right_value) in parameter:
#        for (left_length, rigth_length) in lenght:
#
#            (left_model, left_error,
#             right_model, right_error) = model.call(input_signal=input_signal,
#                                                    left_value=left_value,
#                                                    left_length=left_length,
#                                                    right_value=right_value,
#                                                    rigth_length=rigth_length,
#                                                    peak_signal=peak_signal)
#            if final_left_error > left_error:
#                final_left_model = left_model
#                final_left_error = left_error
#                final_left_value = left_value
#                final_left_length = left_length
#
#            if final_right_error > right_error:
#                final_right_model = right_model
#                final_right_error = right_error
#                final_right_value = right_value
#                final_right_length = rigth_length
#
#    results = dict()
#    results['final_left_model'] = final_left_model
#    results['final_left_error'] = final_left_error
#    results['final_left_value'] = final_left_value
#    results['final_left_length'] = final_left_length
#    results['final_right_model'] = final_right_model
#    results['final_right_error'] = final_right_error
#    results['final_right_value'] = final_right_value
#    results['final_right_length'] = final_right_length
#
#    return results

#def get_automatic_parameters(function, packets_of_signals,
#                             list_of_evaluetor, list_of_peaks=None, n_tests=1):
#    if n_tests < packets_of_signals:
#        raise ValueError("n_tests parameter mus be smallest"
#                         + "than or equal len(packets_of_signals)")
#    perm = np.random.permutation(len(packets_of_signals))[:n_tests]
#    evalueted_signals = evaluete_models(function, packets_of_signals,
#                                        list_of_evaluetor, list_of_peaks)
#    
    
    
    