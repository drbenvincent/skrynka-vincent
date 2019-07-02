import numpy as np
from scipy.stats import cauchy
from scipy.optimize import minimize



class Model:

    def calc_aic(self):
        return -2*self.ll + 2*self.free_params

    def calc_bic(self):
        # NOTE THAT THE NUMBER OF OBSERVATIONS = n_bootstrap_samples
        return -2*self.ll + np.log(n_bootstrap_samples)*self.free_params

    def fit(self):
        '''Find parameters which minimise the negative log likelihood.'''
        if not self.bounds:
            result = minimize(self.nll, self.x0,
                              method='Nelder-Mead',
                              options={'disp': True})
        else:
            result = minimize(self.nll, self.x0,
                              method='L-BFGS-B', bounds=self.bounds,
                              options={'disp': True})
        self.mlparams = result.x
        self.nll = result.fun
        self.ll = -self.nll
        self.AIC = self.calc_aic()
        self.BIC = self.calc_bic()
        return self


class H1(Model):
    """Our control (trait-only) model which assumes zero change in AUC"""

    name = "1. Trait only"
    x0 = [0.05]
    free_params = len(x0)
    # bounds = [(0., None)]
    bounds = None

    @staticmethod
    def nll(params):
        return (-sum(cauchy.logpdf(data['delta_food'], loc=0, scale=params[0]) +
                     cauchy.logpdf(data['delta_money'], loc=0, scale=params[0]) +
                     cauchy.logpdf(data['delta_music'], loc=0, scale=params[0])))


class H2(Model):
    """In-domain model"""

    name = "2. In-domain"
    x0 = [-0.25, 0.05]
    free_params = len(x0)
    # bounds = [(None, 0.), (0., None)]
    bounds = None

    @staticmethod
    def nll(params):
        return -sum(cauchy.logpdf(data['delta_food'], loc=params[0], scale=params[1]) +
                   cauchy.logpdf(data['delta_money'], loc=0, scale=params[1]) +
                   cauchy.logpdf(data['delta_music'], loc=0, scale=params[1]))


class H3(Model):
    """Monetary primacy model"""

    name = "3. Monetary primacy"
    x0 = [-0.25, 0.05]
    free_params = len(x0)
    # bounds = [(None, 0.), (0., None)]
    bounds = None

    @staticmethod
    def nll(params):
        return -sum(cauchy.logpdf(data['delta_food'], loc=params[0], scale=params[1]) +
               cauchy.logpdf(data['delta_money'], loc=params[0], scale=params[1]) +
               cauchy.logpdf(data['delta_music'], loc=0, scale=params[1]))


class H4(Model):
    """Devaluation model"""

    name = "4. Devaluation"
    x0 = [-0.25, 0.1, 0.05]
    free_params = len(x0)
    bounds = [(None, 0.), (0., None), (0., None)]

    @staticmethod
    def nll(params):
        return -sum(cauchy.logpdf(data['delta_food'], loc=params[0], scale=params[2]) +
               cauchy.logpdf(data['delta_money'], loc=params[1], scale=params[2]) +
               cauchy.logpdf(data['delta_music'], loc=params[1], scale=params[2]))


class H5(Model):
    """Spillover model"""

    name = "5. Spillover"
    x0 = [-0.25, -0.1, 0.05]
    free_params = len(x0)
    # bounds = [(None, 0.), (None, 0.), (0., None)]
    bounds = None

    @staticmethod
    def nll(params):
        return -sum(cauchy.logpdf(data['delta_food'], loc=params[0], scale=params[2]) +
               cauchy.logpdf(data['delta_money'], loc=params[1], scale=params[2]) +
               cauchy.logpdf(data['delta_music'], loc=params[1], scale=params[2]))


class H6(Model):
    """State-only model"""

    name = "6. State-only"
    x0 = [-0.25, 0.05]
    free_params = len(x0)
    # bounds = [(None, 0.), (0., None)]
    bounds = None

    @staticmethod
    def nll(params):
        return -sum(cauchy.logpdf(data['delta_food'], loc=params[0], scale=params[1]) +
               cauchy.logpdf(data['delta_money'], loc=params[0], scale=params[1]) +
               cauchy.logpdf(data['delta_music'], loc=params[0], scale=params[1]))
