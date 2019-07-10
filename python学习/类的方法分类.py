# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:22:28 2019

@author: Administrator
"""

class Pizza(object):
    radius = 42
    def __init__(self, size=10):
        self.size = size
    def get_size(self):
        return self.size
    @staticmethod
    def mix_ingredients(x, y):
        return x + y 
    def cook(self):
        return self.mix_ingredients(self.cheese, self.vegetables)
    @classmethod
    def get_radius(cls):
        return cls.radius

Pizza.get_size() #报错，因为是实例函数，需要实例化
Pizza.get_size(Pizza())
Pizza(42).get_size()  #两种方式都可以调用实例函数

Pizza.get_radius() #类方法是可以直接调用的

Pizza.mix_ingredients(1,3)
Pizza().mix_ingredients(1,3) #静态方法类和实例都可以直接调用