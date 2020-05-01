# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:10:59 2020

@author: vedenev
"""

import numpy as np
import ctypes as ct
import cv2

N_CANNELS = 3

def flatten_for_cuda(image: np.ndarray) -> np.ndarray:
    # y, x, c - > c , y, x
    return np.transpose(image,  axes=(2, 0, 1)).flatten()

def unflatten_from_cuda(image_flattened: np.ndarray, height: int, widht: int) -> np.ndarray:
    return np.transpose(image_flattened.reshape(N_CANNELS, height, widht), axes=(1, 2, 0))
    
def get_blur():
    dll = ct.windll.LoadLibrary("./blur.dll") 
    func = dll.blur
    # paramenters: image, images_size_x, images_size_y, kernel_size, sigma, image_out
    func.argtypes = [ct.POINTER(ct.c_ubyte), ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_float, ct.POINTER(ct.c_ubyte)] 
    return func

__blur = get_blur()

def blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    
    image_size_x = image.shape[1]
    image_size_y = image.shape[0]
    
    image_out_size_x = image_size_x - kernel_size + 1;
    image_out_size_y = image_size_y - kernel_size + 1;
    image_out_size = image_out_size_x * image_out_size_y * N_CANNELS;
    
    image_flatten = flatten_for_cuda(image)
    image_flatten_pointer = image_flatten.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    image_out_flatten = np.zeros(image_out_size, np.uint8)
    image_out_flatten_pointer = image_out_flatten.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    
    __blur(image_flatten_pointer, image_size_x, image_size_y, kernel_size, sigma, image_out_flatten_pointer)
    
    image_out = unflatten_from_cuda(image_out_flatten, image_out_size_y, image_out_size_x)
    return image_out

if __name__ == '__main__':
    image = cv2.imread(r'img.jpg')	
    
    image_out = blur(image, 15, 3)
    cv2.imwrite('img_output_cuda.png', image_out)
    np.save('img_output_cuda.npy', image_out)
    