#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:33:56 2019

@author: chenxiahang
"""

class Solution(object):
    def longestPalindrome(self,s):
        size = len(s)
        if size <= 1:
            return s
        dp = [[False for _ in range(size)] for _ in range(size)]
        longest_l = 1
        res = s[0]
        for r in range(1, size):
            for l in range(r):
                if s[l] == s[r] and (r - l <= 2 or dp[l + 1][r - 1]):
                    dp[l][r] = True
                    cur_len = r - l + 1
                    if cur_len > longest_l:
                        longest_l = cur_len
                        res = s[l:r + 1]
        print(res)
    
s = "abcdzdcab"
lp = Solution()
lp.longestPalindrome(s)