# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:34:40 2019

@author: Administrator
"""

#np.pad使用https://blog.csdn.net/qq_34650787/article/details/80500407

import numpy as np
A = np.arange(95,99).reshape(2,2) 

np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))
np.pad(A,((3,2),(2,3)),'constant',constant_values = (-2,2))

np.pad(A,((3,2),(2,3)),'constant',constant_values = ((0,0),(1,2))) 

np.pad(A,((3,2),(2,3)),'constant') #缺省表示填充0


#边缘填充
B = np.arange(1,5).reshape(2,2) 
np.pad(B,((1,2),(2,1)),'edge')