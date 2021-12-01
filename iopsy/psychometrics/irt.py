from numpy import exp
from pandas import DataFrame

def rasch_irf(theta, b):
    """Rasch Item Response Function
    
    Given a level of theta and an item difficulty return the probability of a correct response.
    
    Parameters
    ----------
    theta : numeric
        the person's theta (ability) level
    b : numeric
        the item difficulty level
    """
    return(exp(theta - b)/(1 + exp(theta - b)))

def generate_random_rasch_items(theta, bs = None):
    """Generate Random Items
    
    This is useful when you just need some placeholder data that behaves like real data.
    
    Parameters
    ----------
    theta : iterable collection of numerics
        the thetas to generate item responses relative to
    bs : iterable collection of numerics
        for every level of b provided, an item will be generated with that difficulty parameter
    """
    if bs is None:
        bs = [x/2 for x in range(-6,7)]
    probs = {}
    for b in bs:
        probs[f'item_{b}'] = rasch_irf(theta, b)
    df = DataFrame(probs)
    return df.applymap(lambda p: rd.choices([0,1], [1-p, p])[0])