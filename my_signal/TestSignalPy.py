# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 08:51:13 2019

@author: Weslley L. Caldas
"""
from scipy import signal
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from scipy.misc import electrocardiogram
import matplotlib.pyplot as plt
import SignalWave

import unittest
 
class TestSignalPy(unittest.TestCase):
    def setUp(self):
        self.ecg = electrocardiogram()
        self.ecg = self.ecg - np.mean(self.ecg)
        self.ecg = self.ecg[100:150]
        m = SignalWave.MexicanHat()
 
    def test_type_response(self):
        '''
        Non-marked lines should only get 'p' tags around all input
        '''
        self.assertEqual(type(m.call(self.ecg,5)),type(int))
 

if __name__ == '__main__':
    unittest.main()
 