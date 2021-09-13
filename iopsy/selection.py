import numpy as np

class WeightedLassoRegression:
    def __init__(self, alpha = 0, alpha_weights = None):
        """Weighted Lasso Regression

        This is a regression algorithm that includes an additional penalizer in its cost function. The penalty
        is similar to that of lasso regression, using the sum of the absolute values of the models weights as an
        added penalty to the cost function. However, this penalty is modified by a set of weights, which allows
        for including outside information into the regression. By increasing these weights the model can be 
        discouraged from weighting specific variables too strongly. 

        By adjusting the alpha weights, a model can be discouraged from placing weight on variables with known
        issues, such as large group differences, or temporal instability. This relies on prior knowledge of the
        variables characteristics. Because the information used to regularize the model is not neccessary at the 
        time of prediction it does not amount to race norming or other illegal (in the US) forms of scoring. 
        Allowing for a model that is optimized for accuracy while de-weighting potentially problematic variables
        from overly-influencing the model.

        Algorithmic hiring law in the US is still fairly new and subject to change. This was built in an effort 
        to be compliant with US law, but also with the social good.

        Parameters
        ----------
        alpha : numeric
            The overall weight of the regularization
        alpha_weights : collection of numeric
            The weights of each predictor variable used in regularization. Larger weights discourage the use of
            the corresponding variable (e.g., when fitting a model with 4 X variables, a weighting of [1, 1, 5, 1]
            would discourage the model from placing much weight on the third variable
        """
        self.alpha = alpha
        self.alpha_weights = alpha_weights
    
    def fit(self, X, y):
        """Fit
        
        Fit the model using the predictors specified in X and the criterion specified in y
        
        Parameters
        ----------
        X : k x n Array of Data
        y : 1 x n Array of Data
        """
        from scipy.optimize import minimize
        xdim = np.shape(X)
        
        coefs = [0]
        coefs.extend(np.zeros(xdim[1]))
        
        if self.alpha_weights is None:
            alpha_weights = [1]*xdim[1]
        else:
            alpha_weights = self.alpha_weights
        
        def cost(coefs):
            b = coefs[0]
            weights = coefs[1:]
            resid = np.sum(np.abs(y - np.dot(X, weights) - b))
            penalty = np.sum(np.abs(alpha_weights * weights)) * xdim[0]
            return resid + self.alpha * penalty
        
        res = minimize(cost, coefs)
        self.coefs = res['x']
        
    def predict(self, X):
        """Predict
        
        Once the model has been fit, this will make predictions using the trained model weights
        
        Parameters
        ----------
        X : k x n Array of Data
        """
        return(np.dot(X, self.coefs[1:]) + self.coefs[0])
    
    def score(self, X, y):
        """Score
        
        Once the model has been fit, use the provided data to score the model on the RMSE, MAE and corr
        
        Parameters
        ----------
        X : k x n Array of Data
        y : 1 x n Array of Data
        """
        yhat = self.predict(X)
        resid = y-yhat
        res = {'rmse': np.sqrt(np.mean(resid**2)),
               'mae': np.mean(np.abs(resid)),
               'r': np.corrcoef(y, yhat)[0,1]}
        return res