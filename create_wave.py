# lstm autoencoder recreate sequence
import numpy as np
import pandas as pd
from matmodel import utils, feature_builder as fb
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, lfilter
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout



def butter_bandpass(lowcut, highcut, fs, order=5, analog=True):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=analog)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, analog=True):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, analog=analog)
    y = lfilter(b, a, data)
    return y


dataset = np.load('./data/morfology.npy').item()
n = 600
fs = 360
list_ecg = dataset['signal'][:n]
list_peaks = dataset['peaks'][:n]
list_labels = dataset['label'][:n]
list_labels, b = pd.factorize(list_labels)
list_of_packets = []
for (sig, peak, label) in zip(list_ecg, list_peaks, list_labels):
    sig == np.subtract(sig, np.mean(sig))
    list_of_packets.append(fb.LabeledPacket(np.array(sig), peak, label))

my_ecg = fb.ECG.build_from_packets(list_of_packets)
len_packet = 35
list_ecg = my_ecg.build_packets(len_packet)
list_ecg = [x.input_signal for x in my_ecg.packets]
df = pd.DataFrame(list_ecg).values
df = df.T


















#filters = dict()
#filters['T'] = (.1, .20)
#filters['P'] = (.15, .30)
#filters['QRS'] = (.3, .5)
#filters['ALL'] = (.70, .9)
#
#analog_filters = dict()
#analog_filters['T'] = (0, 10)
#analog_filters['P'] = (5, 30)
#analog_filters['QRS'] = (8, 50)
#analog_filters['ALL'] = (70, fs)
#
#
#analog = False
#processed = []
#for key in filters:
#    lowcut, highcut = filters[key]
#    processed.append(butter_bandpass_filter(df, lowcut, highcut,
#                                            fs, order=5, analog=analog))
#
#
#analog = True
#processed = []
#for i in range(1, 7):
#    lowcut, highcut = i/10, i/10+0.2
#    print(lowcut)
#    processed.append(butter_bandpass_filter(df, fs*lowcut, fs*highcut,
#                                            fs, order=5, analog=analog))
#
#
#for p in processed[1:]:
#    plt.plot(p[:, 1])
#    plt.show()
