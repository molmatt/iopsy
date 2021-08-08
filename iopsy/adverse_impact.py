from analysis import Analysis

class AdverseImpact(Analysis):
    def __init__(self, data, cutscore, x, y = None, filters = None, referent = None):
        super().__init__(data, 
                         analysis = 'AdverseImpact', 
                         x = x, y = y, 
                         filters = filters)
        self.cutscore = cutscore
        if referent is not None:
            self.referent = referent
        else:
            self.referent = referent
            
def cut(y, score = None, groups = None):
    if score is not None:
        return(y >= score)
    elif groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        return(y.isin(groups))
    
def selection_rates(score, by):
    df = pd.concat([score, by], axis = 1)
    res = df.groupby(by).agg(['mean', 'count'])
    res.columns = ['sr', 'n']
    return res

def determine_referent(sr, min_n = 5):
    sr = sr[sr['n'] >= min_n].copy()
    return(sr.sort_values(['sr','n'], ascending = False).index[0])