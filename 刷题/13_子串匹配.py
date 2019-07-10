# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:25:37 2019

@author: Administrator
"""


class Solution:
    """
    @param source: 
    @param target: 
    @return: return the index
    """
    def strStr(self, source, target):
        n1 = len(source)   
        n2 = len(target)
        if n1 < n2:
            return -1
        if source is None or target is None:
            return -1
        for i in range(n1-n2+1):
            j=0
            while(j < n2):
                if source[i+j] != target[j]:
                    break
                j += 1
            if j == n2:
                return i
        return -1
    
    
Solution().strStr('abcdabcdefg','bcde')
    
        