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
        for col, filts in filters.items():
            if not isinstance(filts, list):
                filts = [filts]
            for filt in filts:
                self.data = drop_outlier(self.data, col, **filt)
        self.p = None
        self.effect = None
        
def cat_count(x):
    """Category Count
    
    Calculate the count of each category and return a series of the same length as x with
    these counts as values in place of the corresponding category. This can be used with 
    filtering to remove less prevalent groups, or conversely to limit your search to only them.
    
    Parameters
    ----------
    x : pandas.Series
    """
    vc = x.value_counts()
    return(x.replace(vc.index,vc))

def cat_prop(x):
    """Category Proportion
    
    Calculate the proportion of each category and return a series of the same length as x with
    these proportions as values in place of the corresponding category. This can be used with 
    filtering to remove less prevalent groups, or conversely to limit your search to only them.
    
    Parameters
    ----------
    x : pandas.Series
    """
    props = x.value_counts(normalize = True)
    return(x.replace(props.index, props))

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

def drop_outlier(data, x, method = standard_score, val = 2, less = False, **kwargs):
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
    less : bool
        Should the value to drop be below the critical value?
    """
    if less:
        bool_mask = method(data[x], **kwargs).abs() > val
    else:
        bool_mask = method(data[x], **kwargs).abs() < val
    return(data[bool_mask])
