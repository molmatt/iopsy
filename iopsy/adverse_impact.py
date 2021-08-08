from analysis import Analysis

class AdverseImpact(Analysis):
    def __init__(self, data, x, y = None, cutscore = None, groups = None, filters = None, 
                 referent = None, min_ref = 5):
        super().__init__(data, 
                         analysis = 'AdverseImpact', 
                         x = x, y = y, 
                         filters = filters)
        self.cutscore = cutscore
        self.groups = groups
        self.score = cut(data[y], cutscore = cutscore, groups = groups)
        self.selection_rates = selection_rates(self.score, data[x])
        
        if referent is not None:
            self.referent = referent
        else:
            self.referent = determine_referent(self.selection_rates, min_ref = 5)
            
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

def determine_referent(sr, min_ref = 5):
    sr = sr[sr['n'] >= min_ref].copy()
    return(sr.sort_values(['sr','n'], ascending = False).index[0])