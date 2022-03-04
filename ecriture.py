import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy.fft as fft
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import integrate
import numpy as np
import os
import quat
import vector
import rotmat


def window(a, w = 128, o = 64, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    

def diffT(df, w):
    Tmin = df[0]
    Tmax = df[w]
    diffT = df[w] - df[0]
    return diffT


def transform(window, diffT):
    #Calcul FFT
    Fwindow = 1/(diffT) * fft.fft(window, axis=1) 
    return Fwindow

def reTime(df):
	df['time'] = df['time'] - df['time'][0]

def rePressure(df):
	df['pressure'][df['pressure'] <0] += 1
	

def ecriture(df):
	df['normXY'] = (df['x']**2 + df['y']**2).apply(math.sqrt)
	win = window(df['pressure'],128,64, copy=True)
	winxy = window(df['normXY'],128,64,copy=True)
	diff = diffT(df['time'],128)
	trans = transform(win,diff)
	transXY = transform(winxy,diff)
	transShifted = fft.fftshift(trans)
	transXYShifted = fft.fftshift(transXY)
	return transXYShifted, transShifted

def decoupe(df):
	listeMot =[]

	#############Reperage des coupures dans les données ##########
	for i in range(1,len(tsa1['x'])):
    		if ((tsa1['x'][i] - tsa1['x'][i-1] > 2) or (tsa1['x'][i] - tsa1['x'][i-1] < -2)):
        		listeMot.append(i)

	#############Création des df en fonction des coupures (listeMot) ##############
	############# Ajout des noms des df dans "listeDF"               ##############
	df0=tsa1.iloc[:listeMot[0]]
	listeDF=['df0']
	for i in range(1,len(listeMot)):
    		globals()['df'+str(i)]= tsa1.iloc[listeMot[i-1]:listeMot[i]]
    		listeDF.append('df' + str(i))
	n = len(listeMot)
	globals()['df' + str(n)]=tsa1.iloc[listeMot[n -1]:]
	listeDF.append('df' + str(n))
	

	###########SUPPRESSION DES DF AYANT UNE LONGUEUR TROP COURTE POUR ETRE UN MOT#############
	for i in range(0,len(listeDF)):
    		if len(globals()['df'+str(i)]) <100:
           		listeDF.remove('df' + str(i))
    
    