# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:11:52 2020

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt

image_pytorch = np.load('img_output.npy')
image_cuda = np.load('img_output_cuda.npy')

difference = image_cuda.astype(np.int16) - image_pytorch.astype(np.int16)

difference_flattened = difference.flatten()

min_t = np.min(difference_flattened)
max_t = np.max(difference_flattened)
mean_t = np.mean(difference_flattened)
std_t = np.std(difference_flattened)

print("min_t =", min_t)
print("max_t =", max_t) 
print("mean_t = ", mean_t)
print("std_t =", std_t)

#min_t = -5
#max_t = 7
#mean_t =  0.7363226188483967
#std_t = 0.6413344010889628

plt.plot(difference_flattened, 'k.')
plt.xlabel('element No.')
plt.ylabel('difference')