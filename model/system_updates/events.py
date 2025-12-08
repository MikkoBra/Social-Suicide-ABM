import numpy as np


def event_happens(p):
    """
    Simulate an external strategy event happening with given
    probability.
    """
    return np.random.uniform() <= p


def apply_event(val, prob, strength, weight):
    """
    Apply the effect of a single event occurring.

    Parameters
    ----------
    prob: float
        Probability of the event occurring in a single timestep
    strength: float
        Absolute value representing the effect of the event
    weight: float
        Weight that the "strength" parameter is modified by
        through multiplication
    """
    if event_happens(p=prob):
        new_val = val + strength * weight
        print(f"Event occurred, old value: {val}, new value: {new_val}")
        return new_val
    return val
