# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:28:08 2020

@author: vedenev
"""
import numpy as np

#a = np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

a1 = np.asarray([[1,2,3], [4,5,6]])
a2 = np.asarray([[7,8,9], [10,11,12]])
a3 = np.asarray([[13,14,15], [16,17,18]])

a = np.stack((a1, a2, a3), axis=2)

# y, x, c - > c , y, x

h = a.shape[0]
w = a.shape[1]

af = np.transpose(a,  axes=(2, 0, 1)).flatten()

ar = np.transpose(af.reshape(3, h, w), axes=(1, 2, 0))

res = np.all(ar == a)
print('res =', res)