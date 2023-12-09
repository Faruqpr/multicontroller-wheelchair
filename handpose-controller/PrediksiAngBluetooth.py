# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:25:32 2022

@author: eko my
"""

import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

import ModulEkstraksiFiturBluetooth as md 

# Bluetooth Library
#import bluetooth
import bluetooth
import time

Kelas =["Stop","Maju","Kiri","Kanan","Mundur"]
md.Deteksi2(0,"cAng.h5",Kelas,[md.cAng,md.cOrientasi], imsize= (320, 240))
