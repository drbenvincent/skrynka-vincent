import pymc3 as pm
import numpy as np
import math
import numpy.matlib
from sklearn.metrics import log_loss, roc_auc_score


delays = np.linspace(0, 365, 500)


def df_to_vars(data):
    """Extract data from dataframe into separate variables"""
    RA = data.A.values
    DA = data.DA.values
    RB = data.B.values
    DB = data.DB.values
    R = data.R.values
    return (RA, DA, RB, DB, R)


# CORE MODEL FUNCTIONS


def choice_psychometric(VA, VB, α, ϵ=0.01):
    return ϵ + (1.0-2.0*ϵ) * cumulative_normal((VB-VA)/α)


def cumulative_normal(x):
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))


# ============================================
# MODEL BASE CLASS
# ============================================


class Model:

    def __init__(self, data):
        self.data = data
        self.model = None
        self.trace = None
        self.metrics = None
        self.build_model(self.data)

    def fit(self, sample_options):
        '''Note `sample_options` is a dictionary of options'''
        # sampling
        with self.model:
            self.trace = pm.sample(**sample_options)

        self.waic = pm.waic(self.trace, self.model)
        print(self.waic)
        self.metrics = {'log_loss': self.calc_log_loss(),
                        'AUC': self.calc_AUC(), 
                        'WAIC': self.waic.WAIC, 
                        'roc_auc': self.calc_roc_auc()}

    def calc_log_loss(self):
        """Returns a distribution of log loss scores, one for each sample"""
        R_predicted_prob = self.trace.P_chooseB
        R_actual = self.data['R'].values

        nsamples = R_predicted_prob.shape[0]
        nresponses = R_actual.shape[0]
        assert np.ndim(R_predicted_prob) == 2, "R_predicted is a vector(?) but should be a 2D matrix"
        assert R_predicted_prob.shape[1] == nresponses, "cols in R_predicted should equal number of responses"
        print('Calculating Log Loss metric')
        try:
            R_actual = numpy.matlib.repmat(R_actual, nsamples, 1)
            ll = [log_loss(R_actual[n, :], R_predicted_prob[n, :])
                for n in range(0, nsamples)]
            return ll
        except:
            return np.nan
        
    def calc_roc_auc(self):
        '''Compute Area Under the Receiver Operating Characteristic Curve. 
        We do this for each of the MCMC samples and return the mean of these AUC values'''
        print('Calculating Area Under the Receiver Operating Characteristic Curve')
        try:
            R_predicted_prob = self.trace.P_chooseB
            n_samples = R_predicted_prob.shape[0]
            R_actual = self.data['R'].values
            return np.mean([roc_auc_score(R_actual, R_predicted_prob[i,:])
                            for i in range(n_samples)])
        except: 
            return np.nan
            
    def calc_AUC(self, max_delay=30):
        '''Return the normalised area under curve'''

        # convert mean param _values_ into tuple
        params = self.mean_parameters()
        mean_params = tuple([params[field] for field in self.params])

        delays = np.linspace(0, max_delay, 500)
        
        # send those (unpacked) to the plotting function
        df = self._df(delays, *mean_params)

        normalised_delays = delays / np.max(delays)
        AUC = np.trapz(df, x=normalised_delays)
        return AUC

    def mean_parameters(self):
        '''Return the posterior mean parameter values in a dictionary'''
        params = {param_name: np.mean(self.trace[param_name]) for param_name in self.params}
        return params

    def plot(self, ax, label=None):
        '''Plot the discount function corresponding to posterior mean'''
        # just plot up to the delays we looked at for this data
        max_delays = np.max(self.data['DB'])
        delays = np.linspace(0, max_delays, 500)
        # convert mean param _values_ into tuple
        params = self.mean_parameters()
        mean_params = tuple([params[field] for field in self.params])
        # send those (unpacked) to the plotting function
        df = self._df(delays, *mean_params)
        ax.plot(delays, df, label=label)



# ============================================
# DELAYED CHOICE MODEL: Hyperbolic discounting
# ============================================


class Exponential(Model):

    params = ['k']

    @staticmethod
    def discount_function(delay, k):
        ''' Exponential discounting of time'''
        return pm.math.exp(-k*delay)

    @staticmethod
    @numpy.vectorize
    def _df(delay, k):
        ''' Exponential discounting of time'''
        return math.exp(-k*delay)

    def build_model(self, data):

        (RA, DA, RB, DB, R) = df_to_vars(self.data)

        with pm.Model() as model:

            # Priors
            k = pm.HalfNormal('k', sd=0.5)
            alpha = pm.Exponential('alpha', lam=1)

            # subjective value functions
            VA = pm.Deterministic('VA', RA * self.discount_function(DA, k))
            VB = pm.Deterministic('VB', RB * self.discount_function(DB, k))

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_psychometric(VA, VB, alpha))

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P_chooseB, observed=R)
        self.model = model


class Hyperbolic(Model):

    params = ['logk']

    @staticmethod
    def discount_function(delay, logk):
        ''' Hyperbolic discounting of time'''
        return 1.0 / (1.0+(pm.math.exp(logk)*delay))

    @staticmethod
    @numpy.vectorize
    def _df(delay, logk):
        '''for plotting'''
        k = math.exp(logk)
        return 1.0 / (1.0+(k*delay))

    def build_model(self, data):

        (RA, DA, RB, DB, R) = df_to_vars(self.data)

        with pm.Model() as model:

            # Priors
            logk = pm.Normal('logk', mu=pm.math.log(1/50), sd=2)
            alpha = pm.Exponential('alpha', lam=1)

            # subjective value functions
            VA = pm.Deterministic('VA', RA * self.discount_function(DA, logk))
            VB = pm.Deterministic('VB', RB * self.discount_function(DB, logk))

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_psychometric(VA, VB, alpha))

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P_chooseB, observed=R)
        self.model = model


class ModifiedRachlin(Model):

    params = ['logk', 's']

    @staticmethod
    def discount_function(delay, logk, s):
        ''' Modified Rachlin discounting of time'''
        return 1.0 / (1.0+(pm.math.exp(logk)*delay)**s)

    @staticmethod
    @numpy.vectorize
    def _df(delay, logk, s):
        '''for plotting'''
        k = math.exp(logk)
        return 1.0 / (1.0+(k*delay)**s)


    def build_model(self, data):

        (RA, DA, RB, DB, R) = df_to_vars(self.data)

        with pm.Model() as model:

            # Priors
            logk = pm.Normal('logk', mu=pm.math.log(1/50), sd=2)
            s = pm.HalfNormal('s', sd=0.5)
            alpha = pm.Exponential('alpha', lam=1)

            # subjective value functions
            VA = pm.Deterministic('VA', RA * self.discount_function(DA, logk, s))
            VB = pm.Deterministic('VB', RB * self.discount_function(DB, logk, s))

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_psychometric(VA, VB, alpha))

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P_chooseB, observed=R)
        self.model = model
        
        
class Hyperboloid(Model):

    params = ['logk', 's']

    @staticmethod
    def discount_function(delay, logk, s):
        ''' Modified Rachlin discounting of time'''
        return 1.0 / (1.0+pm.math.exp(logk)*delay)**s

    @staticmethod
    @numpy.vectorize
    def _df(delay, logk, s):
        '''for plotting'''
        k = math.exp(logk)
        return 1.0 / (1.0+k*delay)**s


    def build_model(self, data):

        (RA, DA, RB, DB, R) = df_to_vars(self.data)

        with pm.Model() as model:

            # Priors
            logk = pm.Normal('logk', mu=pm.math.log(1/50), sd=2)
            s = pm.HalfNormal('s', sd=1)
            alpha = pm.Exponential('alpha', lam=1)

            # subjective value functions
            VA = pm.Deterministic('VA', RA * self.discount_function(DA, logk, s))
            VB = pm.Deterministic('VB', RB * self.discount_function(DB, logk, s))

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_psychometric(VA, VB, alpha))

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P_chooseB, observed=R)
        self.model = model