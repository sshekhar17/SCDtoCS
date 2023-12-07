from math import log, sqrt
from collections import deque
from abc import ABC, abstractmethod

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

from utils import ModifiedDeque, createBoundedSource 


class BaseChangeDetector(ABC):
    """
    Base class for change detection schemes.
    Implement specific change detection methods by overriding methods.
    """
    def __init__(self, source, Nmax=1000, alpha=0.005,
                 progress_bar=False, max_num_CSs=1000):
        """
        source: generator function, for creating the stream of observations
        Nmax: int, denotes the maximum horizon to run the scheme
        alpha: float in (0, 1), characterizes the requirement on average run length (ARL)
        progress_bar: boolean, toggles display of a progress bar
        max_num_CSs: maximum number of active CSs to consider in the scheme
        """
        self.source = source
        self.Nmax = Nmax
        self.alpha = alpha
        self.progress_bar = progress_bar
        self.max_num_CSs = max_num_CSs
        self.data = []

    @abstractmethod
    def empty_intersection(self, activeCSs):
        """Return True if the intersection of CSs in activeCSs is empty."""
        pass

    @abstractmethod
    def init_conf_seq(self, n0=1):
        """Start a new confidence sequence (CS) with the n0-th element of self.data."""
        pass

    @abstractmethod
    def update_conf_seq(self, n, state):
        """
        Update the confidence sequence (CS) state.

        Parameters:
        - n: int, current time index
        - state: dict, contains all the information to specify the current state of the CS

        Returns the updated state of the CS.
        """
        pass

    def one_trial(self):
        """
        Run one trial of the change detection experiment.

        Returns:
        - change_detected: bool, True if a change was detected
        - stopping_time: int, the time at which the algorithm stopped
        """
        activeCSs = ModifiedDeque(maxlen=self.max_num_CSs)
        change_detected = False
        source_iterator = self.source()

        for i in range(self.Nmax):
            x = next(source_iterator)
            self.data.append(x)

            for j, cs in enumerate(activeCSs):
                cs_ = self.update_conf_seq(n=i+1, state=cs)
                activeCSs[j] = cs_

            new_cs = self.init_conf_seq(n0=i+1)
            activeCSs.append(new_cs)

            change_detected = self.empty_intersection(activeCSs)
            if change_detected:
                break

        return change_detected, i+1

    def run_expt(self, num_trials=100):
        """
        Run multiple trials of the change detection experiment.

        Parameters:
        - num_trials: int, number of trials to run

        Returns:
        - Flag: list of bool, indicates if a change was detected in each trial
        - StoppingTimes: numpy array, stopping times for each trial
        """
        Flag = [False] * num_trials
        StoppingTimes = np.ones((num_trials,)) * self.Nmax
        range_ = tqdm(range(num_trials)) if self.progress_bar else range(num_trials)

        for trial in range_:
            self.data = []
            Flag[trial], StoppingTimes[trial] = self.one_trial()

        return Flag, StoppingTimes


class BoundedMeanHoeffdingSCD(BaseChangeDetector):
    """
    Sequential change detection (SCD) in means of bounded observations
    using Hoeffding Confidence Sequence (CS).
    """
    def __init__(self, source, Nmax=1000, alpha=0.005,
                 progress_bar=False, max_num_CSs=500,
                 xmin=0, xmax=1):
        super().__init__(source, Nmax, alpha, progress_bar, max_num_CSs)
        self.xmin = xmin
        self.xmax = xmax

    def init_conf_seq(self, n0=1):
        """
        Initialize Hoeffding CS.

        $CS_t = [\mu_t \pm w_t]$
        $\mu_t = \frac{ \sum_{i=1}^t \lambda_i X_i}{\sum_{i=1}^t \lambda_i}$
        $w_t = \frac{\log(2/\alpha) + \sum_{i=1}^t \lambda_i^2/8}{\sum_{i=1}^t \lambda_i}$
        where
        $\lambda_t = \sqrt{\frac{8 \log(2/\alpha)}{t \log(t+1)}}$ for all $t \geq 1$
        """
        if len(self.data) < n0:
            raise Exception(f"Starting point ({n0}) exceeds the data size")

        state = {
            'n0': n0,
            'mu_num': 0,  # numerator of mu
            'lambda_sum': 1,  # sum of lambda values
            'w_num': log(2/self.alpha),  # numerator of width w_t
            'l': self.xmin,
            'u': self.xmax
        }

        return state


    def update_conf_seq(self, n, state):
        """
        One step update of the Hoeffding CS.
        """
        assert n >= len(self.data)
        X_t = self.data[n-1]
        n0 = state['n0']
        t = n - n0 + 1

        lambda_t = sqrt(8 * log(2/self.alpha) / (t * (t + 1)))
        state['lambda_sum'] += lambda_t
        state['w_num'] += (lambda_t * lambda_t / 8)
        state['mu_num'] += (lambda_t * X_t)

        mu_ = (state['mu_num'] / state['lambda_sum'])
        width_ = state['w_num'] / state['lambda_sum']
        l_, u_ = mu_ - width_, mu_ + width_

        state['l'] = max(l_, state['l'])
        state['u'] = min(u_, state['u'])

        return state

    def empty_intersection(self, activeCSs):
        """
        Return True if the intersection of CIs contained in activeCSs is empty.
        """
        l_, u_ = self.xmin, self.xmax

        for state in activeCSs:
            l_ = max(l_, state['l'])
            u_ = min(u_, state['u'])

        return u_ < l_

class BoundedMeanBernsteinSCD(BaseChangeDetector):
    """
    Sequential change detection (SCD) in means of bounded observations
    using Bernstein Confidence Sequence (CS).
    """
    def __init__(self, source, Nmax=1000, alpha=0.005,
                 progress_bar=False, max_num_CSs=500,
                 xmin=0, xmax=1):
        super().__init__(source, Nmax, alpha, progress_bar, max_num_CSs)
        self.xmin = xmin
        self.xmax = xmax

    def init_conf_seq(self, n0=1):
        """
        Initialize Bernstein CS.

        $CS_t = [\mu_t +- w_t]$, where 

        $\hat{\sigma}_t^2 = \frac{1}{t+1}(1/4 + \sum_{i=1}^t (X_i - \hat{\mu}_i)^2 )$
        $ \hat{\mu}_t = \frac{1}{t+1} (1/2 + \sum_{i=1}^t X_i)$

        $\psi_E(\lambda) = (-\log(1-\lambda) - \lambda)/4$ 

        $\lambda_t = \min{3/4, \sqrt{ \frac{2 \log(2/\alpha)}{\hat{\sigma}_{t-1}^2 t \log(t+1)}} }$
        $\mu_t = \frac{\sum_{i=1}^t \lambda_i X_i}{\sum_{i=1}^t \lambda_i}$
        $w_t = \frac{\log(2/\alpha) + 4\sum_{i=1}^t (X_i - \hat{\mu}_{i-1})^2 \psi_E(\lambda_i) }{\sum_{i=1}^t \lambda_i}$
        """
        if len(self.data) < n0:
            raise Exception(f"Starting point ({n0}) exceeds the data size")

        state = {
            'n0': n0,
            'muhat_num': 1/2,  # numerator of \hat{\mu}_t
            'sig_num': 1/4,  # numerator of \hat{\sigma}_t^2
            'mu_num': 0,  # numerator of mu_t
            'lambda_sum': 1,  # sum of lambda values
            'w_num': log(2/self.alpha),  # numerator of width w_t
            'l': self.xmin,
            'u': self.xmax
        }

        return state

    def psiE(self, lambda_):
        """
        Compute psi_E(lambda).
        """
        assert lambda_ < 1
        return (-log(1 - lambda_) - lambda_) / 4

    def update_conf_seq(self, n, state):
        """
        One step update of the Bernstein CS.
        """
        assert n >= len(self.data)
        X_t = self.data[n-1]
        n0 = state['n0']
        t = n - n0 + 1

        sig_hat_sq = state['sig_num'] / t
        mu_hat = state['muhat_num'] / t

        state['sig_num'] += (X_t - mu_hat)**2
        state['muhat_num'] += X_t

        lambda_t = min(3/4, sqrt(2 * log(2/self.alpha) / (sig_hat_sq * t * (t + 1))))
        state['lambda_sum'] += lambda_t
        state['w_num'] += 4 * ((X_t - mu_hat)**2) * self.psiE(lambda_t)
        state['mu_num'] += (lambda_t * X_t)

        mu_ = (state['mu_num'] / state['lambda_sum'])
        width_ = state['w_num'] / state['lambda_sum']
        l_, u_ = mu_ - width_, mu_ + width_

        state['l'] = max(l_, state['l'])
        state['u'] = min(u_, state['u'])

        return state

    def empty_intersection(self, activeCSs):
        """
        Return True if the intersection of CIs contained in activeCSs is empty.
        """
        l_, u_ = self.xmin, self.xmax

        for state in activeCSs:
            l_ = max(l_, state['l'])
            u_ = min(u_, state['u'])

        return u_ < l_

