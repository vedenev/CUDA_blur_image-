# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:58:42 2020

@author: vedenev
"""

import numpy as np
import ctypes as ct
import cv2
import sys
import time


class Blurer:
    def __init__(self, image_size_x: int, image_size_y: int , kernel_size:int, sigma: float):
        
        self.N_CANNELS = 3
        
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        self.image_out_size_x = self.image_size_x - self.kernel_size + 1;
        self.image_out_size_y = self.image_size_y - self.kernel_size + 1;
        self.image_out_size = self.image_out_size_x * self.image_out_size_y * self.N_CANNELS;
        
        is_windows = sys.platform.startswith('win')
        if is_windows:
            self.shared_library = ct.windll.LoadLibrary("./blurer.dll") 
        else:
            self.shared_library = ct.cdll.LoadLibrary('./blurer.so')
        
        self.shared_library.Blurer_new.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_float] 
        self.shared_library.Blurer_new.restype = ct.c_void_p
        
        self.shared_library.Blurer_blur.argtypes = [ct.c_void_p, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_ubyte)] 
        self.shared_library.Blurer_blur.restype = ct.c_void_p
        
        self.blurer_object = self.shared_library.Blurer_new(image_size_x, image_size_y, kernel_size, sigma)
        
        self.image_out_flatten = np.zeros(self.image_out_size, np.uint8)
        self.image_out_flatten_pointer = self.image_out_flatten.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    
    def _flatten_for_cuda(self, image: np.ndarray) -> np.ndarray:
        # y, x, c - > c , y, x
        return np.transpose(image,  axes=(2, 0, 1)).flatten()
    
    def _unflatten_from_cuda(self, image_flattened: np.ndarray, height: int, widht: int) -> np.ndarray:
        return np.transpose(image_flattened.reshape(self.N_CANNELS, height, widht), axes=(1, 2, 0))
    
    def blur(self, image: np.ndarray) -> np.ndarray:
        
        image_flatten = self._flatten_for_cuda(image)
        image_flatten_pointer = image_flatten.ctypes.data_as(ct.POINTER(ct.c_ubyte))
        
        
        self.shared_library.Blurer_blur(self.blurer_object, image_flatten_pointer, self.image_out_flatten_pointer)
        image_out = self._unflatten_from_cuda(self.image_out_flatten, self.image_out_size_y, self.image_out_size_x)
        return image_out

if __name__ == '__main__':
    
    image_size_x = 1200
    image_size_y = 800
    kernel_size = 15
    sigma = 3.0
    blurer = Blurer(image_size_x, image_size_y, kernel_size, sigma)
    
    image = cv2.imread(r'img.jpg')	
    
    n_repeats = 1000
    time_1 = time.time()
    for repeat_index in range(n_repeats):
        image_out = blurer.blur(image)
    time_2 = time.time()
    time_mean = (time_2 - time_1) / n_repeats
    print('time_mean =', time_mean)
    # time_mean = 0.02526753902435303
    # GPU: Nvidia GTX760, 2GB
    # GPU load: 73%
    # GPU memory: 35 MB
    cv2.imwrite('img_output_cuda_test_speed.png', image_out)
    np.save('img_output_cuda_test_speed.npy', image_out)