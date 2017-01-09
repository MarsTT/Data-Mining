# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 22:46:40 2017

@author: beta
"""

#x = int(raw_input("Please enter a number: "))
x=int(raw_input("Please enter a number: "))
if x < 0:
   x = 0
   print 'Negative changed to zero'
elif x == 0:
     print 'Zero'
elif x == 1:
     print 'Single'
else:
     print 'More'