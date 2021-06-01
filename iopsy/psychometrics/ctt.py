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
