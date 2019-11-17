# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:53:58 2019

@author: Weslley L. Caldas
"""


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matmodel import utils, feature_builder as fb
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matmodel import utils, feature_builder as fb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from scipy import stats
import copy
from sklearn.metrics import mean_squared_error as mse

def normalize_amplitude(orig_signal, ref_signal):
        xmax = np.max(orig_signal)
        xmin = np.min(orig_signal)
        ymax = np.max(ref_signal)
        ymin = np.min(ref_signal)
        if xmax == xmin:
            return orig_signal
        else:
            return ((ymax-ymin)/(xmax-xmin))*(orig_signal-xmin)+ymin

def restore_model(name):
    return fb.deserialize(utils.read_file(name))

def smooth(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def polinomy(weights, x):
    new_x = []
    for xx in x:
        i = len(weights)-1
        value = 0
        for w in weights: 
            value += w * xx**i
            i = i-1
            
        new_x.append(value)
    return new_x


def polinomy_integrate(x):
    new_x = []
    weights = [9.89452693e-08, -1.77367719e-06, -1.29136853e-07, 1.16019276e-04,
        3.24784480e-04, -8.30166136e-03, -3.03896626e-02, 7.65295510e-01,
       -3.96644146e+00, 9.15129978e+00, -9.58408684e+00, 3.52778690e+00,
        1.45125655e-01]

    i = len(weights)-1
    value = 0
    for w in weights: 
        value += w * x**i
        i = i-1
        
    return value



def polinomy_text(weights, n=7):
    new_x = []
    i = len(weights)-1
    value = ""
    for w in weights[:-1]: 
        value += str(truncate(w, n)) + '*x^' + str(i) + " "
        i = i-1
    value += str(w)

    return value

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def load_mathematical(name):
    model = restore_model(name)
    table = model.get_errors_table(normalized=False)
    table = MinMaxScaler().fit_transform(table.T)
    table = pd.DataFrame(table.T)
    table['is_valey'] = [int(x.is_valey) for x in model.packets]
    return table


noises = [0, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

color = ['b-', 'r-', 'g-', 'y-', 'c-', 'm-', 'k-', 'v-']

Names = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']

results = [list() for x in range(4)]
for j, r in enumerate([0, 200, 400, 600]):
    print('Noise:' + str(r))
    for i, n in enumerate(noises):
        model = load_mathematical('m_model_' + str(n))
        result = model.values[r:r+200,:10].mean(axis=0)
        results[j].append(result)
        plt.plot(model.values[r:r+200,:10].mean(axis=0), color[i])
    plt.legend(noises)
    plt.title(Names[j], loc='left')
    plt.show()
    plt.close()
l=[]
for j in [0,1,2,3]:
    for i in range(200):
        l.append(j)

kl = scipy.stats.entropy

t = model.values[r:r+200,:10].mean(axis=0)
x = [x for x in range(0, len(t))]
z = np.polyfit(range(0, len(t)), t, 12)
plt.plot(polinomy(z,x))

noises = [0, 0.1, 0.2, 0.3, 0.4]
re = []
for j, r in enumerate([16]):
    for i, n in enumerate(noises):
        m = restore_model('m_model_' + str(n) + '_real')
        plt.plot(m.packets[r].input_signal)
        re.append(m.packets[r].input_signal)
    plt.legend(noises)
    plt.show()
    plt.close()

a = matrix_to_tikz(re, 
                   [np.linspace(0, 0.1388888888888889,50) for x in range(50)],
                   [RMnoises])

#
#from collections import Counter
#x = model.values[r:r+200,1]
#C = Counter(x)
#total = float(sum(C.values()))
#for key in C: C[key] /= total
#
#
#counts, bins = np.histogram(x,bins=20)
#bins = bins[:-1] + (bins[1] - bins[0])/2
#
#print (np.trapz(counts, bins))
#from scipy.stats import norm
#
#binwidth = (max(data)-min(data))/20
#plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), density=True)




def pdf(x, a, b, c, d, e, f, p):
    if x < d:
        return pdf_trap(x, a, b, c, d)*p
    else:
        return pdf_triangle(x, d, e, f)*(1-p)


def pdf_trap(x, a, b, c, d):
    mu = 2/(d+c-b-a)
    if (a <= x) & (x < b):
        return (mu*(x-a))/(b-a)
    if (b <= x) & (x < c):
        return mu
    if (c <= x) & (x < d):
        return (mu*(d-x))/(d-c)
    return 0


def pdf_triangle(x, a, b, c):
    if (x < 0):
        return 0
    if (a <= x) & (x < c):
        return (2*(x-a))/((b-a)*(c-a))
    if (c == x):
        return 2/(b-a)
    if (c < x) & (x <= b):
        return (2*(b-x))/((b - a)*(b - c))
    if (b < x):
        return 0


m_model = restore_model('m_model_0')

(list_of_signals, list_of_models, list_of_names, list_of_errors, list_of_types) = m_model.get_signals_and_models(normalized=True)


input_signal = m_model.packets[0].input_signal[:28]


a = 0
b = a
c = 3.5
d = 6.5
e = 10
f = e
p = 0.35

for c in [2, 2.5, 3.5, 4, 5, 6, ]:
    for d in np.linspace(1, 3, 3):
        y = []
        x = np.linspace(a, f, 28)
        for i in x:
            y.append(pdf(i, a, b, c, d+c, e, f, p))
#            y.append(pdf_triangle(i, 0, 1, 1))
        plt.plot(x,y)
#        plt.show()
#        plt.close()
        yy = normalize_amplitude(input_signal,y)
        plt.plot(x, yy)
        plt.title(kl(yy, y))
        plt.show()
        plt.close()
#input_signal = model.packets[200].input_signal
#x = [x for x in range(0, len(input_signal))]
#z = np.polyfit(range(0, len(input_signal)), input_signal, 100)
#plt.plot(polinomy(z,x))
#input_signal = smooth(input_signal,10)
#plt.plot(input_signal)
#(cA, cD) = pywt.dwt(input_signal, 'db1')
#coeff = pywt.wavedec(input_signal, 'db1', level=2)
#plt.plot(coeff[2])
#plt.plot(cA)
#plt.plot(cD)







#def plot(x):
#    plt.plot(x)
#    plt.show()
#    plt.close()
#
#def rayleigh_func(x, n, scale):
#    return (x/(scale**2)) * np.e**((-x**2)/2*scale**2)
#
#def gaussian_func(x, n, sigma):
#    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp((-(x-0)**2)/2*sigma**2)
#
#x = np.linspace(-1, 1, 100)
#x2 = np.linspace(0, 2, 100)
#for i in range(3, 12):
#    a = gaussian_func(x, len(x), i)
##    plot(a)
#    b = rayleigh_func(x2, len(x), i)
##    plot(b)
#    plot(a-b)
#    plot(b-a)
#    plot(a+b)
#    plot(b+a)
#    plot(b/a)
#    plot(a/b)

