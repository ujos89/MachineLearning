import numpy as np
import matplotlib.pyplot as plt

def AND(x1, x2):
	w1, w2, theta = .5, .5, .7
	tmp = w1*x1 + w2*x2
	if(tmp>theta):
		return 1
	else:
		return 0
'''
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
'''


