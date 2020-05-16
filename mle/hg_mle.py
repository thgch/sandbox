"""
hg_mle.py

Author: thgch
Created on 2020/05/14

"""

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
from math import pi
from scipy.optimize import minimize
# import pandas as pd


class AbsMle(metaclass=ABCMeta):

    def __init__(self, dist_name, dim_of_param, param_label, series_samples=None):
        self._dist_name = dist_name
        self._dim_of_param = dim_of_param
        self._param_label = copy.deepcopy(param_label)
        self._params = np.empty(dim_of_param, dtype=np.float64)
        self._series_samples = copy.deepcopy(series_samples)

    @property
    def distribution_type(self):
        return self._dist_name

    @property
    def params(self):
        return self._params

    @property
    def samples(self):
        return self._series_samples

    @params.setter
    def params(self, params):
        self._params = copy.deepcopy(params)

    @samples.setter
    def samples(self, samples):
        self._series_samples = copy.deepcopy(samples)

    def likelihood(self, params):
        return np.exp(self.log_likelihood(params))

    @abstractmethod
    def log_likelihood(self, params):
        pass

    def estimate_params(self):
        obj_func = lambda params: -self.log_likelihood(params)
        init = np.ones(self._dim_of_param)
        result = minimize(obj_func, x0=init, method="BFGS")
        self._params = copy.deepcopy(result.x)
        return self._params


class Normal1DMle(AbsMle):

    def __init__(self, series_samples=None):
        super().__init__('Normal-1D', 2, series_samples)

    def log_likelihood(self, params):
        samples = super().samples
        n_samples = len(samples)
        mean, stdv = params[0], params[1]
        varn = stdv * stdv
        res = -0.5 * n_samples * np.log(2.0 * pi * varn) - 0.5 * (np.power(samples - mean, 2)).sum() / varn
        return res


class Cev1DMle(AbsMle):

    def __init__(self, series_samples=None):
        super().__init__('CEV-1D', 2, series_samples)

    def log_likelihood(self, params):
        samples = super().samples
        n_samples = len(samples)
        mean, stdv = params[0], params[1]
        varn = stdv * stdv
        res = -0.5 * n_samples * np.log(2.0 * pi * varn) - 0.5 * (np.power(samples - mean, 2)).sum() / varn
        return res
