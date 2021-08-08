from .analysis import Analysis

class AdverseImpact(Analysis):
    """Adverse Impact Analysis
    
    An overarching class to handle adverse impact analysis workflows.
    
    Parameters
    ----------
    data : pandas.DataFrame
    x : str column name
        The column name of the categorical variable across which to look for adverse impact.
        This is typically a demographic variable indicating a protected class (e.g., 'gender',
        'ethnicity').
    y : str column name
        The column name of the variable containing the score upon which hiring decisions will
        be made. If the column is numeric, a cutscore must be specified. If the column is 
        categorical, a string or list of strings must be specified as the passing group(s).
    cutscore : numeric
        The cutscore to be used for deciding what is passing and what is failing. Any number 
        that is this number or greater will be considered a pass, anything less than that will
        be a fail. This must be specified if y indicates a numeric variable.
    groups : string or list of strings
        The groups that should be counted as passing. Any group that isn't specified here will
        be coded as a fail.
    referent : string or None
        If specified this forces this group to serve as the referent against which all other
        groups will be compared. If left as None this will be inferred from the data using the
        logic specified in determine_referent. Essentially, highest passing group with some 
        additional considerations for small sample sizes and ties.
    min_ref : int
        The minimum sample size a group must have in order to be considered for election as 
        referent
    """
    def __init__(self, data, x, y, cutscore = None, groups = None, filters = None, 
                 referent = None, min_ref = 5):
        super().__init__(data, 
                         analysis = 'AdverseImpact', 
                         x = x, y = y, 
                         filters = filters)
        self.cutscore = cutscore
        self.groups = groups
        self.score = cut(data[y], score = cutscore, groups = groups)
        self.selection_rates = selection_rates(self.score, data[x])
        if referent is not None:
            self.referent = referent
        else:
            self.referent = determine_referent(self.selection_rates, min_ref = min_ref)
        self.effect = impact_ratios(self.selection_rates, referent = self.referent)
        self.p = fet_series(data[x], self.score, self.referent)
        
    def summary(self):
        from pandas import concat
        return(concat([self.selection_rates, self.effect, self.p], axis = 1))
            
def cut(y, score = None, groups = None):
    """Implement a Cutscore
    
    Implement a cutscore on a variable according to the numeric score set in score, or the string
    or list of strings specified in groups. 
    
    Parameters
    ----------
    y : pandas.Series
        The series containing the score upon which hiring decisions will
        be made. If the series is numeric, a cutscore must be specified. If the series is 
        categorical, a string or list of strings must be specified as the passing group(s).
    cutscore : numeric
        The cutscore to be used for deciding what is passing and what is failing. Any number 
        that is this number or greater will be considered a pass, anything less than that will
        be a fail. This must be specified if y indicates a numeric variable.
    groups : string or list of strings
        The groups that should be counted as passing. Any group that isn't specified here will
        be coded as a fail.
        
    Returns
    -------
    pandas.Series of Booleans with True denoting passing, and False denoting failing
    """
    if score is not None:
        return(y >= score)
    elif groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        return(y.isin(groups))
    
def selection_rates(score, by):
    """Calculate Selection Rates
    
    Calculate the selection rates and sample size for each group specified in the by variable.
    Selection rates refer to the proportion of individuals who pass.
    
    Parameters
    ----------
    score : pandas.Series of Booleans
        A series of booleans indicating whether the individual passed (True) or failed (False)
    by : pandas.Series of string
        The categorical variable by which to group the individuals. This is most typically gender
        or ethnicity.
        
    Returns
    -------
    pandas.DataFrame with groups as index and selection rates (sr) and sample size (n) as columns
    """
    from pandas import concat
    df = concat([score, by], axis = 1)
    res = df.groupby(by).agg(['mean', 'count'])
    res.columns = ['sr', 'n']
    return res

def determine_referent(sr, min_ref = 5):
    """Determine the Referent
    
    From the output of selection_rates, determine the referent group. This is defined as the
    highest passing group with a sample size that is comprised of at least the min_ref. This is to
    prevent small, statistically unstable groups from being determined the referent. In the event of
    a tie, the group with the largest sample size will be chosen.
    
    Parameters
    ----------
    sr : pandas.DataFrame
        The output of selection_rates. A dataframe with the demographic groups as the index, and two
        columns, sr (selection rates) and n (sample size).
    min_ref : int
        the smallest sample size to still be considered for election to referent.
        
    Returns
    -------
    String name of the referent group
    """
    sr = sr[sr['n'] >= min_ref].copy()
    return(sr.sort_values(['sr','n'], ascending = False).index[0])

def impact_ratios(sr, referent):
    """Calculate Impact Ratios
    
    Impact ratios are the measure of the 4/5ths rule. Essentially it is the ratio of every groups 
    selection rate relative to the selection rate of the referent group. If this resulting ratio
    of ratios is less than 4/5ths (.8) then this is considered evidence of adverse impact.
    
    Parameters
    ----------
    sr : pandas.DataFrame
        The output of selection_rates. A dataframe with the demographic groups as the index, and two
        columns, sr (selection rates) and n (sample size).
    referent : string
        Name of the group that is serving as the referent.
    """
    ir = sr['sr']/sr.loc[referent,'sr']
    ir.name = 'ir'
    return ir

def contingency_generator(by, score, referent):
    """Contingency Table Generator
    
    This is a generator that yields a 2x2 contingency table on every iteration. This will provide a
    table for every other groups comparison with the referent group.
    
    Parameters
    ----------
    by : pandas.Series of string
        The variable to group the data by. In the context of adverse impact this is a demographic 
        variable of some sort.
    score : pandas.Series of Booleans
        A series of booleans indicating whether the individual passed (True) or failed (False).
    referent : string
        Name of the group that is serving as the referent.
    """
    from pandas import crosstab
    tab = crosstab(by, score)
    for focal in tab.index.drop(referent):
        yield(focal, tab.loc[[focal, referent]])

def fet_series(by, score, referent):
    """Fisher Exact Test Series
    
    Perform a series of Fisher Exact Tests of each group relative to the referent group. 
    
    Parameters
    ----------
    by : pandas.Series of string
        The categorical variable by which to group the individuals. This is most typically gender
        or ethnicity.
    score : pandas.Series of Booleans
        A series of booleans indicating whether the individual passed (True) or failed (False)
    referent : string
        Name of the group that is serving as the referent.
        
    Returns
    -------
    pandas.Series of p values from the fisher exact tests.
    """
    from scipy.stats import fisher_exact
    from pandas import Series
    idx, pval = [], []
    for focal, tab in contingency_generator(by, score, referent):
        idx.append(focal)
        pval.append(fisher_exact(tab)[1])
    return(Series(pval, index = idx, name = 'fet_p'))
