class Analysis:
    """Analysis
    
    This is a generic base class for building more specific analyses from.
    """
    def __init__(self, data, analysis, x, y = None, filters = None):
        self.data = data
        self.analysis = analysis
        self.x = x
        self.y = y
        self.filters = filters
        self.p = None
        self.effect = None
        
def drop_small_n(data, x, n = None, prop = None):
    vc = data[x].value_counts()
    if n is not None:
        keep = vc[vc > n].index
    elif prop is not None:
        props = vc/vc.sum()
        keep = vc[props > prop].index
    return data[data[x].isin(keep)]

def hampel_identifier(x):
    from scipy.stats import median_abs_deviation
    return((x-x.median())/median_abs_deviation(x, scale = 'normal'))

def standard_score(x):
    return((x-x.mean())/x.std())

def iqr_score(x, q = .25):
    scale = x.quantile(1-q)-x.quantile(q)
    return(x-x.median()/scale)
