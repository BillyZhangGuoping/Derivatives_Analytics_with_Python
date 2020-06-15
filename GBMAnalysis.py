#
# Analyzing Returns from Geometric Brownian Motion
# 03_stf/GBM_returns.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'


#
# Helper Function
#
def dN(x, mu, sigma):
	''' Probability density function of a normal random variable x.
	概率密度函数，等同于cumulative 为false的excel函数 NORM.DIST(x,mean,standard_dev,cumulative)
	Parameters
	==========
	mu: float
	expected value
	sigma: float
	standard deviation
	Returns
	=======
	pdf: float
	value of probability density function
	'''
	z = (x - mu) / sigma
	pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
	return pdf


#
# Simulate a Number of Years of Daily Stock Quotes
#
if __name__ == '__main__':
	print(dN(0.65, 0.6, 0.89))
