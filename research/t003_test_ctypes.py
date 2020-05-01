# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:10:59 2020

@author: vedenev
"""

import numpy as np
import ctypes as ct


def get_cuda_square():
	dll = ct.windll.LoadLibrary("./blur_kernel.dll") 
	func = dll.cudaSquare
	func.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_size_t] 
	return func

__cuda_square = get_cuda_square()

def cuda_square(a, b, size):
	a_p = a.ctypes.data_as(ct.POINTER(ct.c_float))
	b_p = b.ctypes.data_as(ct.POINTER(ct.c_float))

	__cuda_square(a_p, b_p, size)

if __name__ == '__main__':
	size = int(1024) 

	a = np.arange(1, size + 1).astype('float32')
	b = np.zeros(size).astype('float32')

	cuda_square(a, b, size)

	#for i in range(size):
   #print(b[i], end = "")
	#	print( '\t' if ((i % 4) != 3) else "\n", end = " ", flush = True)