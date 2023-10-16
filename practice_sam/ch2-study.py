#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:41:50 2023

@author: sl
"""

import pyautogui as pg 
import time 
from datetime import datetime


print(pg.position())

# Point(x=-892, y=-67)

while True : 
    pg.moveTo(-892,-67)
    pg.click()
    time.sleep(6)