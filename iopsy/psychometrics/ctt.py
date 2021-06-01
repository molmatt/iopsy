def cronbachs_alpha(data):
    """Cronbach's Alpha
    
    Cronbach's alpha is one of the most widely adopted estimates of a scale's inter-item reliability.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    k = data.shape[1]
    sum_item_var = data.var().sum()
    scale_var = data.sum(axis = 1).var()
    return (k/(k-1))*(1-(sum_item_var/scale_var))

def citr(data):
    """Corrected Item-Total Correlations
    
    Calculates an items correlation with the sum of the rest of the items. The correction refers to the
    removal of the item when calculating the sum to correct for an overestimate of the correlation due to
    auto-correlation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    from numpy import corrcoef
    from pandas import Series
    res = []
    for col, df in iterdrop(data):
        x = df.sum(axis = 1)
        y = data[col]
        res.append(np.corrcoef(x, y)[0,1])
    return Series(res, index = data.columns)
        
def iterdrop(data):
    """Iteratively Drop Columns From Data
    
    This is a generator that yields the name of the column dropped as well as the DataFrame missing the
    named column.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df with columns to iteratively drop through
    """
    for col in data.columns:
        yield(col, data.drop(col, axis = 1))
        
def alpha_if_deleted(data):
    """Cronbach's Alpha if Item Deleted
    
    The Cronbach's Alpha if the item were to be deleted. This is meant to assist in the scale refinement
    when seeking to improve reliability. This seems to be largely popularized by its inclusion in SPSS.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    from pandas import Series
    res = []
    for col, df in iterdrop(data):
        res.append(cronbachs_alpha(df))
    return Series(res, index = data.columns)

def item_loadings(data):
    """Items' Loadings on to a Single Factor
    
    Calculate all of the item loadings on to a single factor. This relies heavily on FactorAnalyzer. Its
    really just a wrapper to handle the little things.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    from factor_analyzer.factor_analyzer import FactorAnalyzer
    from pandas import Series
    fa = FactorAnalyzer(n_factors=1, rotation = None)
    fa.fit(data)
    ldng = [val[0] for val in fa.loadings_]
    return Series(ldng, index = data.columns)

def mean_if_deleted(data):
    """Mean Scale Mean if Item is Deleted
    
    The mean scale mean if the item is deleted.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    from pandas import Series
    res = []
    for col, df in iterdrop(data):
        res.append(df.mean(axis = 1).mean())
    return Series(res, index = data.columns)

def sd_if_deleted(data):
    """Standard Deviation of Scale Mean if Item is Deleted
    
    The standard deviation of the scale mean if the item is deleted.
    
    Parameters
    ----------
    data : pandas.DataFrame
        df containing the item responses for the scale
    """
    from pandas import Series
    res = []
    for col, df in iterdrop(data):
        res.append(df.mean(axis = 1).std())
    return Series(res, index = data.columns)

def ctt_item_stats(data):
    """Calculate the Typical Classical Test Theory Item Statistics
    
    Runs a battery of classical test theory item statistics. Caclulates the item mean, item sd,
    scale mean's mean if item deleted, scale mean's sd if item deleted, corrected item total 
    correlation, item loadings on factor, and cronbach's alpha if item deleted.
    
    Parameters
    ----------
    data : pd.DataFrame
        df containing the item responses for the scale
    """
    from pandas import DataFrame
    analyses = {
        'mean': data.mean(),
        'sd': data.std(),
        'mean_if_deleted': mean_if_deleted(data),
        'sd_if_deleted': sd_if_deleted(data),
        'citr': citr(data),
        'loadings': item_loadings(data),
        'alpha_if_deleted': alpha_if_deleted(data)
    }
    return DataFrame(analyses)

