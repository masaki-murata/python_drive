#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 23:05:17 2018

@author: muratamasaki
"""

def func(x):
    y = x % 2
    print(x/y)
    print(x/y)
    
for i in range(10):
    try:
        func(i)
    except:
        print("skip %d" % i)