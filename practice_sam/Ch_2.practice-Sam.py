#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:49:40 2023
핸즈온 머신러닝 3판 2
@author: sl
"""

#%%
#Scikit Learn 버전 확인해보기 

import pandas as pd 

# 이미 테이터를 다운 받았음 

housing = pd.read_csv('./datasets/housing/housing.csv')


#%%

import os

print(os.getcwd())


