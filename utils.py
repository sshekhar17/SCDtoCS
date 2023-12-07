from collections import deque 
from itertools import count 

import numpy as np 


class ModifiedDeque(deque):
    """
    A modified deque container that also allows for 
    indexed access of its elements. 

    This will be used for storing the states of the 
    k-most-recent 'active CSs' by the main change
    detection scheme.   
    """
    def __init__(self, iterable=[], maxlen=None):
        super().__init__(iterable, maxlen=maxlen)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)  
        elif index < 0:
            return self[len(self) + index]
        else:
            return super().__getitem__(index)

    def __setitem__(self, index, value):
        if index < 0:
            index = len(self) + index
        super().__setitem__(index, value)


# Data Source generators

def createBoundedSource(xmin=0.0, xmax=1.0, change_point=100, mu0=0.5, mu1=0.7):
    """
    Return a source that generates observations supported on [xmin, xmax],
    whose mean is "mu0" prior to "change_point", and is "mu1" after that.

    Generate random variables using scaled and shifted beta distributions with
    parameters (a=2, b), where b is chosen to satisfy the mean restrictions:
        mu0 = xmin + (xmax-xmin)*(2/(2+b0)), and
        mu1 = xmin + (xmax-xmin)*(2/(2+b1))
    """
    # Sanity check to avoid boundary conditions
    assert xmin < min(mu0, mu1) <= max(mu0, mu1) < xmax

    # Calculate parameters
    b0 = 2 * (xmax - xmin) / (mu0 - xmin) - 2
    b1 = 2 * (xmax - xmin) / (mu1 - xmin) - 2

    # Define the required source
    def source():
        for i in count():
            if i < change_point:
                x_ = np.random.beta(a=2, b=b0)
            else:
                x_ = np.random.beta(a=2, b=b1)
            x = xmin + (xmax - xmin) * x_
            yield x

    return source

def createUnivariateGaussianSource(change_point=100, mu0=0.0, sig0=1.0, mu1=0.5, sig1=1.0):
    """
    Create a source generating N(mu0, sig0^2) observations prior to
    "change_point", and N(mu1, sig1^2) observations after that.
    """
    def source():
        for i in count():
            if i < change_point:
                x = sig0 * np.random.randn() + mu0
            else:
                x = sig1 * np.random.randn() + mu1
            yield x

    return source


def evaluateExpt(change_point, Flag, StoppingTimes): 
    """
    evaluate the performance of an SCD method

    Parameters:
    -change_point: int, denoting the time at which distribution changed
    -Flag: list of bools, indicates if a change was detected in each trial
    -StoppingTimes: numpy array, containing the stopping times for each trial

    Returns:
    -AvgDetectionDelay: float, average detection delay over trials when Flag==True
    -RejectionRate: float in (0,1), indicating the fraction of trials when Flag==True

    Note:
    -RejectionRate approximates Probability of False alarms if there is no change
    """
    RejectionRate = sum(Flag)/len(Flag) 
    RejectedTimes = StoppingTimes[Flag] 
    if len(RejectedTimes)==0: 
        AvgDetectionDelay = float('inf')
    else:
        AvgDetectionDelay = RejectedTimes.mean()
    return AvgDetectionDelay, RejectionRate


if __name__=='__main__': 
    temp = createBoundedSource()
    source = temp()
    X = []
    for _ in range(1000): 
        x = next(source) 
        X.append(x)
    X = np.array(X) 
    print(f"pre-change mean: {X[:100].mean()}, \t post-change mean: {X[100:].mean()}")
