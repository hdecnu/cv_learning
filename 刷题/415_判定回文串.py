#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:54:34 2019
考虑问题：
1、空字符串和单个字符串
2、去除各种符号，只保留字母和数字，字母统一改为小写
3、看对称的两边值是否相同


@author: chenxiahang
"""


class Solution(object):
    def isPalindrome(self,s):
        alnum = [t.lower() for t in s if t.isalnum()]
        n = len(alnum)
        if n <= 1:
            return True
        mid = n//2
        for i in range(mid):
            if alnum[i] != alnum[n-1-i]:
                return False
        return True
        
s = "race a car"  
pal = Solution()
print(pal.isPalindrome(s))      
