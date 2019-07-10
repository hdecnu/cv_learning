# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:29:23 2019

@author: Administrator
"""
import collections
import string
import random

class Solution:
    def longestPalindrome(self, s):
        ans = 0
        for value in collections.Counter(str).values():
            ans += value//2*2
            if ans%2 == 0 and value%2 == 1:
                ans += 1
        print(ans)
    
        
str = random.choices(list(string.ascii_letters),k=50)  #有放回随机抽样，需要转化为list，k不能省略，sample是无放回抽样
collections.Counter(str)
Solution().longestPalindrome(str)