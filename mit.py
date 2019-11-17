# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:03:37 2019

@author: Weslley L. Caldas
"""

from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from matmodel import utils, feature_builder as fb
import pandas as pd
import wfdb

ListName = ['qRs_4', 'QS_5', 'rRs_8', 'RS_9']

#1) Os 20 primeiros batimentos do sinal 103 - canal 1 (morfologia qRs);
#
#2) Os 20 primeiros batimentos do sinal 100 - canal 2 (morfologia qRs);
#
#3) Os 15 primeiros batimentos do sinal 106 - canal 1 (morfologia qRs);
#
#4) Os 15 primeiros batimentos do sinal 106 - canal 2 (morfologia RS);
#
#5) Os 11 primeiros batimentos do sinal 108 - canal 2 (morfologia QS);
#
#6) Os 20 primeiros batimentos do sinal 111 - canal 2 (morfologia QS);
#
#7) Os 20 primeiros batimentos do sinal 116 - canal 2 (morfologia QS);
#
#8 )Os 20 primeiros batimentos do sinal 121 - canal 2 (morfologia QS);
#
#9) Os 20 primeiros batimentos do sinal 123 - canal 2 (morfologia RS);
#
#10) Os 20 primeiros batimentos do sinal 215 - canal 1 (morfologia RS);
#
#11) Do 10º ao 20º batimento do sinal 217 - canal 1(morfologia RS);
#
#12)  Os 20 primeiros batimentos do sinal 219 - canal 1 (morfologia qRs);
#
#13)  Os 20 primeiros batimentos do sinal 219 - canal 2 (morfologia RS);
#
#14)  Os 20 primeiros batimentos do sinal 223 - canal 2 (morfologia QS);

list_of_signals = [103, 100, 106, 106, 108, 111, 116, 121, 123, 215, 217, 219, 219, 223]
list_of_channels = [1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]

mat = loadmat('data/morfologies_2019.mat')
mat['cell_data']['beats']
mat['cell_data']['peaks']
mat['cell_data']['signal']
mat['cell_data']['channel']
mat['cell_data']['beat_number']
mat['cell_data']['morfology']


my_ecg = fb.ECG(input_signal=filtered, indicators=mat['cell_data']['peaks'],
                        labels=mat['cell_data']['morfology'], is_filtered=False)

