# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:50:18 2019

@author: Weslley L. Caldas
"""
import numpy as np
import multiprocessing as mp
from matmodel import feature_builder as fb
#from bayes_opt import BayesianOptimization


def consume_process(args):
    print("child")
    return consume_evaluetor(args[0], args[1], args[2], args[3], args[4])


def utility_function(m, input_signal, peak_signal):
    
    def black_box_function(a,b,c,d,e,f):
        """Function with unknown internals we wish to maximize.
    
        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        nonlocal m,input_signal, peak_signal
    
        (model, peak_model, errors)= m.call(input_signal=input_signal,
                                values=(a,b), frequency=(c,d), percet_ceif=(e,f),
                                peak_signal=peak_signal,evalue_model=True)
        return -sum(errors)
    return black_box_function

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

def consume_fit_cls(args):
    return args[0].fit(args[1], args[2])

def mp_fit_cls(function, estimators, X, y, workers=2):

    with mp.Pool(processes=workers) as pool:
        result = pool.map(function, [(estimators[index],
                                     X[index], y.copy())
                          for index in range(len(estimators))])
    pool.close()
    return result


def mp_signal_apply(function, packets_of_signals,
                    list_of_evaluetor, workers=2):

    with mp.Pool(processes=workers) as pool:
        result = pool.map(function, [(packets_of_signals[index],
                                     list_of_evaluetor.copy())
                          for index in range(len(packets_of_signals))])
    pool.close()
    return result

def serial_signal_apply(function, packets_of_signals, list_of_evaluetor):
    new_packets_of_signals=[]
    for packet in packets_of_signals:
        print('Passe')
        new_packets_of_signals.append(function(packet, list_of_evaluetor))
    return new_packets_of_signals

def consume_packet_helper(args):
    print('executando')
    return consume_packet(args[0], args[1])


def consume_packet(packet, list_of_evaluetor):
    list_of_error = []
    list_of_parameters = []
    list_of_best_model = []
    list_of_names = []
    for evaluetor in list_of_evaluetor:
        (erros, parameters, best_model) = consume_signal3(packet, evaluetor)
        list_of_error.append(erros)
        list_of_parameters.append(parameters)
        list_of_best_model.append(best_model)
        list_of_names.append(evaluetor.name)
    packet.errors = list_of_error
    packet.parameters = list_of_parameters
    packet.models = list_of_best_model
    packet.names = list_of_names
    return packet

#
#def consume_signal2(packet, evaluetor):
#
#    parameters = dict()
#
#    input_signal = packet.input_signal
#    peak_signal = packet.peak_signal
#    mat_model = evaluetor.mat_model
#
##    for (left_value, right_value) in evaluetor.parameters['values']:
##        for (left_length, rigth_length) in evaluetor.parameters['lenght']:
##            for perc_ceif in evaluetor.parameters['cut']:
##
##                (model, error) = mat_model.call(input_signal=input_signal,
##                                                left_value=left_value,
##                                                left_length=left_length,
##                                                right_value=right_value,
##                                                rigth_length=rigth_length,
##                                                left_ceif=perc_ceif,
##                                                right_ceif=perc_ceif,
##                                                peak_signal=peak_signal)
##                left_model = model[0]
##                left_error = error[0]
##                right_model = model[1]
##                right_error = error[1]
##
##                if final_left_error > left_error:
##                    final_left_model = left_model
##                    final_left_error = left_error
##                    final_left_value = left_value
##                    final_left_length = left_length
##
##                if final_right_error > right_error:
##                    final_right_model = right_model
##                    final_right_error = right_error
##                    final_right_value = right_value
##                    final_right_length = rigth_length
#    
#
#    pbounds = evaluetor.parameters
#
#    
#    optimizer = BayesianOptimization(
#        f=utility_function(mat_model,input_signal,peak_signal),
#        pbounds=pbounds,
#        verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#        random_state=1,
#    )
#    
#    
#    optimizer.maximize(
#        init_points=3,
#        n_iter=3,
#    )
#    
#    params = optimizer.max['params']
#    
#    (best_model, peak_model, errors) = mat_model.call(input_signal=input_signal,
#                                          values=(params['a'],params['b']),
#                                          frequency=(params['c'],params['d']),
#                                          percet_ceif=(params['e'],params['f']),
#                                          peak_signal=peak_signal,
#                                          evalue_model=True)
#   
#
#    best_model = [best_model[:peak_model],best_model[peak_model:]]
#    errors = [errors[0], errors[1]]
#    parameters['values'] = [params['a'], params['c']]
#    parameters['length'] = [params['b'], params['d']]
#
#    return (errors, parameters, best_model)
#


def consume_signal(packet, evaluetor):

    parameters = dict()
    final_left_error = np.inf
    final_right_error = np.inf
    input_signal = packet.input_signal
    len_signal = np.round(len(packet.input_signal)/2)
    peak_signal = packet.peak_signal
    mat_model = evaluetor.mat_model
    parameter = evaluetor.parameters
    is_valey = packet.is_valey

  
    final_left_error=1000000
    final_right_error=1000000
    for values in parameter['values']:
                (model, peak_model, errors) = mat_model.call(input_signal=input_signal,
                                                    values=values,
                                                    frequency=(len_signal,
                                                               len_signal),
                                                    percet_ceif=(0,0),
                                                    peak_signal=peak_signal,
                                                    evalue_model=True,
                                                    is_valey=is_valey)
                
                
                
                left_model = model[:peak_model]
                left_error = errors[0]
                right_model = model[peak_model:]
                right_error = errors[1]
    
                if final_left_error > left_error:
                    final_left_error = left_error
                    final_left_value = values[0]
                    final_left_model = left_model

    
                if final_right_error > right_error:
                    final_right_error = right_error
                    final_right_value = values[1]
                    final_right_model = right_model
    
    final_left_error=1000000
    final_right_error=1000000
    for percet_ceif in parameter['cut']:
                (model, peak_model, errors) = mat_model.call(input_signal=input_signal,
                                                    values=(final_left_value,
                                                            final_right_value),
                                                    frequency=(len_signal,
                                                               len_signal),
                                                    percet_ceif=percet_ceif,
                                                    peak_signal=peak_signal,
                                                    evalue_model=True,
                                                    is_valey=is_valey)
                
                
                
                left_model = model[:peak_model]
                left_error = errors[0]
                right_model = model[peak_model:]
                right_error = errors[1]
    
                if final_left_error > left_error:
                    final_left_model = left_model
                    final_left_error = left_error
                    final_left_ceif = percet_ceif[0]
    
                if final_right_error > right_error:
                    final_right_error = right_error
                    final_right_ceif = percet_ceif[1]
                    final_right_model = right_model

                    
    
    
    final_left_error=1000000
    final_right_error=1000000   
    for frequency in parameter['lenght']:
                (model, peak_model, errors) = mat_model.call(input_signal=input_signal,
                                                    values=(final_left_value,
                                                            final_right_value),
                                                    frequency=frequency,
                                                    percet_ceif=(final_left_ceif,
                                                                 final_right_ceif),
                                                    peak_signal=peak_signal,
                                                    evalue_model=True,
                                                    is_valey=is_valey)
                
                
                
                left_model = model[:peak_model]
                left_error = errors[0]
                right_model = model[peak_model:]
                right_error = errors[1]
    
                if final_left_error > left_error:
                    final_left_model = left_model

                    final_left_error = left_error
                    final_left_freq = frequency[0]
    
                if final_right_error > right_error:
                    final_right_model = right_model

                    final_right_error = right_error
                    final_right_freq = frequency[1]
                    

    
    parameter = dict()
    Values = list(zip(return_range(final_left_value),
                                   return_range(final_right_value)))
    Lenght = list(zip(return_range(final_left_freq),
                                   return_range(final_right_freq)))
    Cut= list(zip(return_range(final_left_ceif),
                                   return_range(final_right_ceif)))
    
    final_left_error=errors[0]
    final_right_error=errors[1]
    for values in Values:
            for frequency in Lenght:
                for percet_ceif in Cut:
    
                    (model, peak_model, errors) = mat_model.call(input_signal=input_signal,
                                                        values=values,
                                                        frequency=frequency,
                                                        percet_ceif=percet_ceif,
                                                        peak_signal=peak_signal,
                                                        evalue_model=True,
                                                        is_valey=is_valey)
                    left_model = model[:peak_model]
                    left_error = errors[0]
                    right_model = model[peak_model:]
                    right_error = errors[1]
    
                    if final_left_error > left_error:
                        final_left_model = left_model
                        final_left_error = left_error
                        final_left_value = values[0]
                        final_left_freq = frequency[0]
                        final_left_ceif= percet_ceif[0]
    
    
                    if final_right_error > right_error:
                        final_right_model = right_model
                        final_right_error = right_error
                        final_right_value = values[1]
                        final_right_freq = frequency[1]
                        final_right_ceif = percet_ceif[1]
    
    (best_model, peak_model2)= mat_model.call(input_signal=input_signal, 
                                 values=(final_left_value,
                                         final_right_value),
                                 frequency=(final_left_freq, 
                                            final_right_freq),
                                 percet_ceif=(final_left_ceif, 
                                              final_right_ceif),
                                 peak_signal=peak_signal)
    
    best_model = [best_model[:peak_model], best_model[peak_model:]]
#    best_model = [final_left_model, final_right_model]
    errors = [final_left_error, final_right_error]
    parameters['values'] = [final_left_value, final_right_value]
    parameters['length'] = [final_left_freq, final_right_freq]
    parameters['cut'] = [final_left_ceif, final_right_ceif]
    return (errors, parameters, best_model)

def consume_evaluetor(input_signal, model, parameter,
                      lenght, peak_signal=None):
    final_left_error = np.inf
    final_right_error = np.inf

    for (left_value, right_value2) in parameter:
        for (left_value2, right_value) in parameter:

            for (left_length, rigth_length) in lenght:
    
                (left_model, left_error,
                 right_model, right_error) = model.call(input_signal=input_signal,
                                                        left_value=left_value,
                                                        left_length=left_length,
                                                        right_value=right_value,
                                                        rigth_length=rigth_length,
                                                        peak_signal=peak_signal,
                                                        is_valey=is_valey)
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



def consume_signal3(packet, mat_model):

    parameters = dict()
    final_left_error = np.inf
    final_right_error = np.inf
    input_signal = packet.input_signal
    peak_signal = packet.peak_signal
    input_values = mat_model.parameters['values']
    input_lenght = mat_model.parameters['lenght']
    input_cut = mat_model.parameters['cut']

    (values, freq, cut) = consume_parameters(packet, mat_model, input_values,
                                               input_lenght, input_cut)
    

    input_values = list(zip(return_range(values[0]),
                        return_range(values[1])))
    input_lenght = list(zip(return_range(freq[0]),
                        return_range(freq[1])))
    input_cut = list(zip(return_range(cut[0]),
                     return_range(cut[1])))

    (values, freq, cut) = consume_parameters(packet, mat_model, input_values,
                                           input_lenght, input_cut)

    (best_model, peak_model, errors)= mat_model.call(input_signal=input_signal, 
                                                     values=(values[0],
                                                             values[1]),
                                                     frequency=(freq[0],
                                                                freq[1]),
                                                     percet_ceif=(cut[0],
                                                                  cut[1]),
                                                     peak_signal=peak_signal,
                                                     evalue_model=True,
                                                     is_valey=packet.is_valey)

    best_model = [best_model[:peak_model], best_model[peak_model:]]
    parameters['values'] = values
    parameters['length'] = freq
    parameters['cut'] = cut
    return (errors, parameters, best_model)


def consume_parameters(packet, mat_model, input_values,
                       input_lenght, input_cut):
    final_left_error = np.inf
    final_right_error = np.inf
    for values in input_values:
            for frequency in input_lenght:
                for percet_ceif in input_cut:
                    (model, peak_model, errors) = mat_model.call(
                     input_signal=packet.input_signal, values=values,
                     frequency=frequency, percet_ceif=percet_ceif,
                     peak_signal=packet.peak_signal, evalue_model=True,
                     is_valey=packet.is_valey)

                    left_error = errors[0]
                    right_error = errors[1]
    
                    if final_left_error > left_error:
                        final_left_error = left_error
                        final_left_value = values[0]
                        final_left_freq = frequency[0]
                        final_left_ceif = percet_ceif[0]
    
    
                    if final_right_error > right_error:
                        final_right_error = right_error
                        final_right_value = values[1]
                        final_right_freq = frequency[1]
                        final_right_ceif = percet_ceif[1]
    output_values = [final_left_value, final_right_value]
    output_freq = [final_left_freq, final_right_freq]
    output_right = [final_left_ceif, final_right_ceif]

    return output_values, output_freq, output_right

    
def return_range(value, percent=5, n=10):
    return np.linspace(value-value/percent,value+value/percent,n)  


def direction_peak(input_signal, factor=0.4):
    id_max = input_signal - np.argmax(input_signal)
    id_min = np.argmin(input_signal)

    if (input_signal[id_max] < np.abs(input_signal[id_min]) * factor):
        return 1
    return 0



def write_file(name, data):
    with open(name, "w") as text_file:
        return text_file.write(data)


def read_file(name):
    with open(name, "r") as text_file:
        return text_file.read()



def get_default_templates(len_packet):
    parameter = dict()
    parameter['values'] = list(zip(np.linspace(1, 15, 30),
                               np.linspace(1, 15, 30)))
    parameter['lenght'] = list(zip(np.linspace(len_packet*0.5, len_packet*2.5, 20),
                               np.linspace(len_packet*0.5, len_packet*2.5, 20)))
    parameter['cut'] = list(zip(np.linspace(0, 60, 7),
                            np.linspace(0, 60, 7)))


    l = []
    for i in np.linspace(1, 15, 30):
        for j in np.linspace(1, 15, 30):
            l.append((i, j))

    r_parameter = dict()
    r_parameter['values'] = l
    r_parameter['lenght'] = list(zip(np.linspace(len_packet*0.5, len_packet*2, 5),
                               np.linspace(len_packet*0.5, len_packet*2, 5)))
    r_parameter['cut'] = list(zip(np.linspace(0, 60, 7),
                                   np.linspace(0, 60, 7)))

    template_gaussian = fb.Gaussian(parameter)
    template_mexican = fb.MexicanHat(parameter)
    template_rayleigh = fb.Rayleigh(r_parameter)
    template_left_rayleigh = fb.LeftInverseRayleigh(parameter)
    template_right_rayleigh = fb.RightInverseRayleigh(parameter)
    
    list_of_templates = [template_gaussian, template_mexican,
                     template_rayleigh, template_left_rayleigh,
                     template_right_rayleigh]

    return list_of_templates
