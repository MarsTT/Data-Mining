# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 00:05:45 2017

@author: beta
"""

number = 10
running = True

while running:
      guess = int(raw_input('Enter an integer : '))
      if guess == number:
         print 'you guess it.' 
         running = False 
      elif guess < number:
           print 'No, it is a little lower' 
      else:
           print 'No, it is a little higher' 
else:
     print 'The while loop is over.' 
     print 'Done' 
3