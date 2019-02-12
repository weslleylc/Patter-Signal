m# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:02:23 2019

@author: Weslley L. Caldas
"""


from bayes_opt import BayesianOptimization
from matmodel import utils, feature_builder as fb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = np.load('./data/morfology.npy').item()
n = 600
list_ecg = dataset['signal'][:n]
list_peaks = dataset['peaks'][:n]
list_labels = dataset['label'][:n]

list_of_packets = []

for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
    list_of_packets.append(fb.LabeledPacket(np.array(sig), peak, label))

my_ecg = fb.LabeledECG.build_from_packets(list_of_packets)
list_ecg = my_ecg.build_packets(35)
list_ecg = my_ecg.packets
one_ecg = list_ecg[1]

m = fb.Rayleigh()
#if one_ecg.input_signal[one_ecg.peak_signal] < 0:
#    one_ecg.input_signal = one_ecg.input_signal*-1
one_ecg.input_signal = MinMaxScaler().fit_transform(one_ecg.input_signal.reshape(-1,1)).reshape(1,-1)[0]

def utility_function(m, input_signal, peak_signal):
    
    def black_box_function(a,b,c=35,d=35,e=0,f=0):
        """Function with unknown internals we wish to maximize.
    
        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        nonlocal m,input_signal, peak_signal
    
        (model, peak_model, errors)= m.call(input_signal=input_signal,
                                values=(a,b), frequency=(c,d), percet_ceif=(e,f),
                                peak_signal=peak_signal,evalue_model=True)
#        plt.plot(model)
#        plt.plot(one_ecg.input_signal)
#        plt.show()
#        plt.close()
        return -sum(errors)
    return black_box_function

# Bounded region of parameter space
#pbounds = {'a': (3, 10), 'b': (3, 10)}

pbounds = {'a': (1, 10), 'b': (1, 10),
           'c': (35, 150), 'd': (35, 150)}

optimizer = BayesianOptimization(
    f=utility_function(m,one_ecg.input_signal,one_ecg.peak_signal),
    pbounds=pbounds,
    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)


optimizer.maximize(
    init_points=50,
    n_iter=5,
)

print(optimizer.max)
params = optimizer.max['params']
for i in range(1,10):
    for j in range(1,10):
        
        (best_model, peak_model, error) = m.call(input_signal=one_ecg.input_signal, 
                                     values=(i,j),
                                     frequency=(35,35),
                                     percet_ceif=(0,0),
                                     peak_signal=one_ecg.peak_signal,
                                     evalue_model=True)
        
        one_ecg.input_signal = one_ecg.input_signal -np.mean(one_ecg.input_signal)
        best_model = best_model -np.mean(best_model)
        plt.plot(best_model)
        plt.plot(one_ecg.input_signal)
        plt.title('Error:' + str(sum(error)))
        plt.show()
        plt.close()