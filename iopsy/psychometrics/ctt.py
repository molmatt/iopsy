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