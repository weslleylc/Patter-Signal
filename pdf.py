from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO 
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import magenta, pink, blue, green
import  numpy as np


c = canvas.Canvas('test.pdf')

 
c.setFont("Courier", 12)
 
form = c.acroForm
x = 25
y = 600
y1, y2, y3, y4, y5, y6, y7 = np.array(range(650, 475, -25)) +150
steep = -5
steep_x = 67
steep_x_column = 100

c.drawString(x, y1, 'QR_1')
form.checkbox(name='cb1', tooltip='Field cb1',
              x=x + steep_x, y=y1+steep, buttonStyle='check', 
              textColor=blue, forceBorder=True)
 
c.drawString(x, y2, 'qR_2')
form.checkbox(name='cb2', tooltip='Field cb2',
              x=x + steep_x, y=y2+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x, y3, 'Qr_3')
form.checkbox(name='cb3', tooltip='Field cb3',
              x=x + steep_x, y=y3+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x, y4, 'qRs_4')
form.checkbox(name='cb4', tooltip='Field cb4',
              x=x + steep_x, y=y4+steep,  buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x, y5, 'QS_5')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x, y=y5+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x, y6, 'R_6')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x, y=y6+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x, y7, 'rR_7')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x, y=y7+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

#####
c.drawString(x+steep_x_column, y1, 'rRs_8')
form.checkbox(name='cb1', tooltip='Field cb1',
              x=x + steep_x+steep_x_column, y=y1+steep, buttonStyle='check', 
              textColor=blue, forceBorder=True)
 
c.drawString(x+steep_x_column, y2, 'RS_9')
form.checkbox(name='cb2', tooltip='Field cb2',
              x=x + steep_x+steep_x_column, y=y2+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x+steep_x_column, y3, 'Rs_10')
form.checkbox(name='cb3', tooltip='Field cb3',
              x=x + steep_x+steep_x_column, y=y3+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x+steep_x_column, y4, 'rS_11')
form.checkbox(name='cb4', tooltip='Field cb4',
              x=x + steep_x+steep_x_column, y=y4+steep,  buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+steep_x_column, y5, 'RSr_12')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+steep_x_column, y=y5+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+steep_x_column, y6, 'rSR_13')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+steep_x_column, y=y6+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+steep_x_column, y7, 'rSr_14')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+steep_x_column, y=y7+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)



#####
c.drawString(x+2*steep_x_column, y1, 'qrS_15')
form.checkbox(name='cb1', tooltip='Field cb1',
              x=x + steep_x+2*steep_x_column, y=y1+steep, buttonStyle='check', 
              textColor=blue, forceBorder=True)
 
c.drawString(x+2*steep_x_column, y2, 'qS_16')
form.checkbox(name='cb2', tooltip='Field cb2',
              x=x + steep_x+2*steep_x_column, y=y2+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x+2*steep_x_column, y3, 'rsRs_17')
form.checkbox(name='cb3', tooltip='Field cb3',
              x=x + steep_x+2*steep_x_column, y=y3+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)
 
c.drawString(x+2*steep_x_column, y4, 'QRs_18')
form.checkbox(name='cb4', tooltip='Field cb4',
              x=x + steep_x+2*steep_x_column, y=y4+steep,  buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+2*steep_x_column, y5, 'Qrs_19')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+2*steep_x_column, y=y5+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+2*steep_x_column, y6, 'qrSr_20')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+2*steep_x_column, y=y6+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)

c.drawString(x+2*steep_x_column, y7, 'Artefato')
form.checkbox(name='cb5', tooltip='Field cb5',
              x=x + steep_x+2*steep_x_column, y=y7+steep, buttonStyle='check',
              textColor=blue, forceBorder=True)


 
fig = plt.figure(figsize=(4, 3))
plt.plot([1,2,3,4])
plt.ylabel('some numbers')

imgdata = BytesIO()
fig.savefig(imgdata, format='png')
imgdata.seek(0)  # rewind the data

Image = ImageReader(imgdata)
c.drawImage(Image, 350, 650, 3.5*inch, 2.5*inch)
c.save()
p.showPage()

