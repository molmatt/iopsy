class Analysis:
    def __init__(self, data, analysis, x, y = None, filters = None):
        """Analysis

        This is a generic base class for building more specific analyses from.
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        self.data = data[x + y]
        self.analysis = analysis
        self.x = x
        self.y = y
        self.filters = filters
        self.p = None
        self.effect = None
        
def drop_small_n(data, x, n = None, prop = None):
    """Drop Small Samples
    
    Drop rows from dataframe based off of the rarity of its value for a given column. This 
    is useful for removing rows that represent fringe cases. The rarity threshold is
    specified by either n, for a raw number, or prop, for a proportion. This function could
    be used to drop rows for any group representing less than 5% of the population. This is
    useful when doing analyses that may be sensitive to small n.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to drop rows from
    x : string column name
        The name of the column to use for determining which rows to drop
    n : numeric or None
        The minimum n of the subgroup to be kept in the dataframe. 
    prop : float between 0 and 1
        The minimum proportion to be kept in the dataframe.
    """
    vc = data[x].value_counts()
    if n is not None:
        keep = vc[vc >= n].index
    elif prop is not None:
        props = vc/vc.sum()
        keep = vc[props >= prop].index
    return data[data[x].isin(keep)]

def hampel_identifier(x):
    """Hampel Identifier
    
    Calculate Hampel Identifiers (HI) for the data. HIs are robust forms of standard score
    (AKA z-scores). A HI is conceptually similar to a Z score but with every instance of 
    mean substituted for median. This prevents the effects of large outliers from biasing
    the estimates.
    
    Parameters
    ----------
    x : pandas.Series
    """
    from scipy.stats import median_abs_deviation
    return((x-x.median())/median_abs_deviation(x, scale = 'normal'))

def standard_score(x):
    """Standard Score
    
    The standard score, z-score, the distance from the mean scaled by the standard deviation.
    This scaling is useful for interpretation purposes as well.
    
    Parameters
    ----------
    x : pandas.Series
    """
    return((x-x.mean())/x.std())

def iqr_score(x, q = .25):
    """Inner Quantile Range Score
    
    The IQR score, the median centered data, scaled by the IQR. The inner quantile range can
    be flexibly set with the q parameter. A q further from .5 implies a larger range, and 
    therefore a smaller IQR Score. 
    
    Parameters
    ----------
    x : pandas.Series
    q : float between 0 and 1
        The quantile value for determining the IQR. A q of .25 yields a quartile
    """
    scale = x.quantile(1-q)-x.quantile(q)
    return(x-x.median()/scale)

def drop_outlier(data, x, method = standard_score, less = False val = 2, **kwargs):
    """Drop Outliers
    
    Drop rows from the dataframe based on whether they are outlying on the column specified
    by x. Outlyingness is determined by the function fed to method, and the critical value
    specified by val.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to drop rows from
    x : string column name
        The name of the column to use for determining which rows to drop
    method : function
        The method to be used for determining outlyingness. Note this requires the actual
        object, not the string of the name.
    val : numeric
        The critical value beyond which any value will be considered outlying.
    """
    if less:
        bool_mask = method(data[x], **kwargs).abs() > val
    else:
        bool_mask = method(data[x], **kwargs).abs() < val
    return(data[bool_mask])
