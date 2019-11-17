# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:36:47 2019

@author: wesll
"""
from mat4py import loadmat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO 
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import magenta, pink, blue, green
import  numpy as np


dataset = loadmat('data/morfologies_2019.mat')
list_ecg = dataset['cell_data']['beats']
list_peaks = dataset['cell_data']['peaks']
list_labels = dataset['cell_data']['morfology']
frequency = 360
const = 1000/frequency
# permutation= np.random.permutation(120)
permutation = [ 97,  45,   9,  79, 104,  82,  95,  20,  98,  77,  24,  61, 102,
        36,  16,  76,  68, 101, 114,  48,  12,  47,  58, 111, 110,  75,
        62, 119,   5,  35,  38,  13,  65,  93,  73,  29,  70,  64,  15,
        87,  40,  49,  66,  53,  27,   2, 107,  41,  94,  26,  55, 117,
        19,  63,  51,  80,  91,  54,  28,  44,  67,  78,  10, 108, 115,
       112,  56,  50,  17,  84,  34,  96,   1, 103, 106,  21,  43,  85,
        92,   7,  46, 116,  99,  69,   3,  52,   6,  14,  39,  71,  89,
         4,  23,  42,  11,  31, 100, 113,  32,  72,  22, 109, 118,  25,
        18,  83, 105,  30,  59,  86,  33,   0,  81,  60,  90,  88,  57,
         8,  74,  37]

c = canvas.Canvas('test.pdf')
c.setFont("Courier", 12)
steep = -5
steep_x = 67
steep_x_column = 100
x = 25
y = 600
aux_page = 1
pass_page = 200
form = c.acroForm

c.drawCentredString(300, 700, 'Basic Information')
c.setFont("Courier", 14)
form = c.acroForm

c.drawString(10, 650, 'First Name:')
form.textfield(name='fname', tooltip='First Name',
               x=110, y=635, borderStyle='inset',
               width=300,
               textColor=blue, forceBorder=True)
 
c.drawString(10, 600, 'Last Name:')
form.textfield(name='lname', tooltip='Last Name',
               x=110, y=585, borderStyle='inset',
               width=300,
               textColor=blue, forceBorder=True)
 
c.drawString(10, 550, 'Address:')
form.textfield(name='address', tooltip='Address',
               x=110, y=535, borderStyle='inset',
               width=400, forceBorder=True)
 
c.drawString(10, 500, 'City:')
form.textfield(name='city', tooltip='City',
               x=110, y=485, borderStyle='inset',
               forceBorder=True)
 
c.drawString(250, 500, 'State:')
form.textfield(name='state', tooltip='State',
               x=350, y=485, borderStyle='inset',
               forceBorder=True)
 
c.drawString(10, 450, 'Zip Code:')
form.textfield(name='zip_code', tooltip='Zip Code',
               x=110, y=435, borderStyle='inset',
               forceBorder=True)
c.showPage()

for i in range(len(permutation)):
    (signal, label) = list_ecg[permutation[i]], list_labels[permutation[i]]

    y1, y2, y3, y4, y5, y6, y7 = np.array(range(1000, 825, -25)) -pass_page*aux_page
    
    
    c.drawString(x, y1, 'QR_1')
    form.checkbox(x=x + steep_x, y=y1+steep, buttonStyle='check', 
                  textColor=blue, forceBorder=True)
     
    c.drawString(x, y2, 'qR_2')
    form.checkbox(x=x + steep_x, y=y2+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x, y3, 'Qr_3')
    form.checkbox(x=x + steep_x, y=y3+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x, y4, 'qRs_4')
    form.checkbox(x=x + steep_x, y=y4+steep,  buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x, y5, 'QS_5')
    form.checkbox(tooltip='Field cb5',
                  x=x + steep_x, y=y5+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x, y6, 'R_6')
    form.checkbox(x=x + steep_x, y=y6+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x, y7, 'rR_7')
    form.checkbox(x=x + steep_x, y=y7+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    #####
    c.drawString(x+steep_x_column, y1, 'rRs_8')
    form.checkbox(x=x + steep_x+steep_x_column, y=y1+steep, buttonStyle='check', 
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+steep_x_column, y2, 'RS_9')
    form.checkbox(x=x + steep_x+steep_x_column, y=y2+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+steep_x_column, y3, 'Rs_10')
    form.checkbox(x=x + steep_x+steep_x_column, y=y3+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+steep_x_column, y4, 'rS_11')
    form.checkbox(x=x + steep_x+steep_x_column, y=y4+steep,  buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+steep_x_column, y5, 'RSr_12')
    form.checkbox(x=x + steep_x+steep_x_column, y=y5+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+steep_x_column, y6, 'rSR_13')
    form.checkbox(x=x + steep_x+steep_x_column, y=y6+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+steep_x_column, y7, 'rSr_14')
    form.checkbox(x=x + steep_x+steep_x_column, y=y7+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    
    
    #####
    c.drawString(x+2*steep_x_column, y1, 'qrS_15')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y1+steep, buttonStyle='check', 
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+2*steep_x_column, y2, 'qS_16')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y2+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+2*steep_x_column, y3, 'rsRs_17')
    form.checkbox(tooltip='Field cb16',
                  x=x + steep_x+2*steep_x_column, y=y3+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
     
    c.drawString(x+2*steep_x_column, y4, 'QRs_18')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y4+steep,  buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+2*steep_x_column, y5, 'Qrs_19')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y5+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+2*steep_x_column, y6, 'qrSr_20')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y6+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    c.drawString(x+2*steep_x_column, y7, 'artifact')
    form.checkbox(x=x + steep_x+2*steep_x_column, y=y7+steep, buttonStyle='check',
                  textColor=blue, forceBorder=True)
    
    
    fig = plt.figure()
    # fig.set_size_inches(2.5, 2)
    plt.plot(const * np.array(range(len(signal))), signal)
    plt.ylabel('Volts')
    plt.xlabel('Time(miliseconds)')
    plt.show()
    # fig.savefig('savefigs/beat{}.png'.format(i), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
    
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)  # rewind the data
    
    Image = ImageReader(imgdata)
    c.drawImage(Image, 350, 850 + - pass_page*aux_page, 3.5*inch, 2.5*inch)
    if aux_page == 4:
        c.showPage()
        aux_page = 1
    else:
        aux_page += 1 
    
c.save()
