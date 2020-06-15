#
# European Call Option Inner Value Plot
# 02_MBV/inner_value_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams['font.family'] = 'serif'
import math
from scipy.integrate import quad


def dN(x):
	''' Probability density function of standard normal random variable x. '''
	return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def N(d):
	''' Cumulative density function of standard normal random variable x. '''
	return quad(lambda x: dN(x), -20, d, limit=50)[0]


def d1f(St, K, t, T, r, sigma):
	''' Black-Scholes-Merton d1 function.
		Parameters see e.g. BSM_call_value function. '''
	d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
	      * (T - t)) / (sigma * math.sqrt(T - t))
	return d1


#
# Valuation Functions
#
def BSM_call_value(St, K, t, T, r, sigma):
	''' Calculates Black-Scholes-Merton European call option value.
	Parameters
	==========
	St : float
		stock/index level at time t
	K : float
		strike price
	t : float
		valuation date
	T : float
		date of maturity/time-to-maturity if t = 0; T > t
	r : float
		constant, risk-less short rate
	sigma : float
		volatility
	Returns
	=======

	call_value : float
		European call present value at t
	'''
	d1 = d1f(St, K, t, T, r, sigma)
	d2 = d1 - sigma * math.sqrt(T - t)
	call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
	return call_value


def BSM_put_value(St, K, t, T, r, sigma):
	''' Calculates Black-Scholes-Merton European put option value.
	Parameters
	==========
	St : float
		stock/index level at time t
	K : float
		strike price
	t : float
		valuation date
	T : float
		date of maturity/time-to-maturity if t = 0; T > t
	r : float
		constant, risk-less short rate
	sigma : float
		volatility
	Returns
	=======
	put_value : float
		European put present value at t

	'''

	put_value = BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T - t)) * K
	return put_value


if __name__ == '__main__':
	# Option Strike
	K = 8000
	# Graphical Output
	S = np.linspace(7000, 9000, 100)  # index level values
	h = np.maximum(S - K, 0)  # inner values of call option
	plt.figure(figsize=(12, 8))
	plt.plot(S, h, lw=2.5)  # plot inner values at maturity
	plt.xlabel('index level $S_t$ at maturity')
	plt.ylabel('inner value of European call option')
	plt.grid(True)
	plt.show()

	# Model and Option Parameters
	K = 8000  # strike price
	T = 1.0  # time-to-maturity
	r = 0.025  # constant, risk-less short rate
	vol = 0.2  # constant volatility

	# Sample Data Generation
	S = np.linspace(4000, 12000, 150)  # vector of index level values
	h = np.maximum(S - K, 0)  # inner value of option
	C = [BSM_call_value(S0, K, 0, T, r, vol) for S0 in S]
	# calculate call option values

	# Graphical Output
	plt.figure(figsize=(12, 8))
	plt.plot(S, h, 'b-.', lw=2.5, label='inner value')
	# plot inner value at maturity
	plt.plot(S, C, 'r', lw=2.5, label='present value')
	# plot option present value
	plt.grid(True)
	plt.legend(loc=0)
	plt.xlabel('index level $S_0$')
	plt.ylabel('present value $C(t=0)$')
	plt.show()
