# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:50:18 2019

@author: Weslley L. Caldas
"""
import numpy as np
import multiprocessing as mp


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

