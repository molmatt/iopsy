class Analysis:
    def __init__(self, data, analysis, x, y = None, filters = None):
        self.data = data
        self.analysis = analysis
        self.x = x
        self.y = y
        self.filters = filters
        self.p = None
        self.effect = None