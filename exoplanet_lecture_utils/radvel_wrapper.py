from copy import deepcopy

import numpy as np
import radvel
from radvel.plot import orbit_plots
from scipy import optimize


class RadvelWrapper:
    """
    A wrapper class for the Radvel package to facilitate the analysis of radial
     velocity data.

    This class provides methods to initialize the model, likelihood, and
    posterior, perform maximum likelihood fitting, plot the radial velocity
    fit, get the posterior results, subtract the fit from the data, and perform
     MCMC fitting.
    """
    def __init__(self, data, guesses):
        """
        Initializes the RadvelWrapper class with the provided data and initial
        guesses.

        Args:
            data (DataFrame): The radial velocity data.
            guesses (list): A list of dictionaries containing initial guesses
            for the parameters. Each guess should contain the keys 'period'
            (period in days), 'kamp' (RV semi-amplitude in m/s), 'emax'
            (maximum eccentricity), 'min_period' (minimum period in days), and
            'max_period' (maximum period in days).
        """
        self.data = data
        self.guesses = guesses

        self._initialize()

    def _bin_same_night(self, rv):
        rv['jd_date'] = rv['time'].apply(lambda x: int(np.floor(x)))
        rv_mean = rv.groupby('jd_date',as_index=False).mean(numeric_only=True)
        rv_n = rv.groupby('jd_date',as_index=False).size()
        rv_mean['errvel'] = rv_mean['errvel'] / np.array(np.sqrt(rv_n['size']))
        return rv_mean

    # Some conveinence functions
    def _initialize_model(self):
        time_base = 2456778
        params = radvel.Parameters(num_planets=len(self.guesses),
                                   basis='per tc secosw sesinw logk')

        for i in range(len(self.guesses)):
            params['per' + str(i + 1)] = radvel.Parameter(
                value=self.guesses[i]['period'], vary=True
            )
            params['logk' + str(i + 1)] = radvel.Parameter(
                value=np.log(self.guesses[i]['kamp']), vary=True
            )

            params['tc' + str(i + 1)] = radvel.Parameter(value=56829.74)
            params['secosw' + str(i + 1)] = radvel.Parameter(value=0.01)
            params['sesinw' + str(i + 1)] = radvel.Parameter(value=0.01)
        params['dvdt'] = radvel.Parameter(value=0)
        params['curv'] = radvel.Parameter(value=0)
        mod = radvel.RVModel(params, time_base=time_base)
        return mod

    def _initialize_likelihood(self, rv, suffix):
        like = radvel.RVLikelihood(
            self.mod, rv.time, rv.mnvel, rv.errvel, suffix=suffix)
        return like


    def _initialize(self):
        rv_series = {}
        for tel in np.unique(self.data['tel']):
            rv_series[tel] = self.data.query("tel == '"+ tel +"'")
            rv_series[tel] = self._bin_same_night(rv_series[tel])

        # t_start = np.min(self.data['time'].values)
        # t_stop = np.max(self.data['time'].values)
        # ti = np.linspace(t_start,t_stop,10000)

        self.mod = self._initialize_model()

        likelihood = {}
        for tel in np.unique(self.data['tel']):
            likelihood[tel] = self._initialize_likelihood(rv=rv_series[tel],
                                                          suffix='_' + tel)
            likelihood[tel].params['gamma_' + tel] = radvel.Parameter(
                value=1.0,
                vary=False,
                linear=True
            )
            likelihood[tel].params['jit_' + tel] = radvel.Parameter(
                value=np.log(1)
            )

        # Build composite likelihood
        like = radvel.CompositeLikelihood(list(likelihood.values()))

        # Set initial values for
        for tel in np.unique(self.data['tel']):
            like.params['jit_' + tel] = radvel.Parameter(value=np.log(2.6))

            # Do not vary dvdt or jitter (Fulton 2015)
            like.params['jit_' + tel].vary = False

        like.params['dvdt'].vary = False
        like.params['curv'].vary = False

        # Instantiate posterior
        self.post = radvel.posterior.Posterior(like)
        self.post0 = deepcopy(self.post)

        # Add in priors
        # Keeps eccentricity < 1
        self.post.priors += [radvel.prior.EccentricityPrior(
            num_planets=len(self.guesses),
            upperlims=[p['emax'] for p in self.guesses]
        )]

        for i in range(len(self.guesses)):
            if ((self.guesses[i]['min_period'] is not None)
                    and (self.guesses[i]['max_period'] is not None)):
                self.post.priors += [radvel.prior.HardBounds(
                    param='per' + str(i + 1),
                    minval=self.guesses[i]['min_period'],
                    maxval=self.guesses[i]['max_period']
                )]

    def max_likelihood_fit(self):
        """
        Performs maximum likelihood fitting on the initialized model and
        updates the posterior.

        This method uses the Powell optimization method to find the parameters
        that maximize the likelihood.
        The initial and final loglikelihoods are printed to the console, along
        with the final posterior.

        No input parameters are required as the method operates on the internal
        state of the object.
        """
        # Perform Max-likelihood fitting
        res = optimize.minimize(
            self.post.neglogprob_array,
            self.post.get_vary_params(),
            method='Powell',
            options=dict(maxiter=100000, maxfev=100000, xtol=1e-8)
        )
        print("Initial loglikelihood = %f" % self.post0.logprob())
        print("Final loglikelihood = %f" % self.post.logprob())
        print(self.post)


    def plot_rv_fit(self):
        """
        Plots the radial velocity fit of the model.

        This method generates a multi-panel plot of the radial velocity fit
        using the current state of the posterior.
        """
        RVPlot = orbit_plots.MultipanelPlot(self.post)
        RVPlot.plot_multipanel()

    def get_posterior(self):
        """
        Returns the posterior results.

        This method returns a list of dictionaries, where each dictionary
        contains the 'kamp', 'period', and 'e' values for each planet in the
        model.

        Returns:
            list of dict: A list of dictionaries containing the 'kamp',
            'period', and 'e' values for each planet.
        """
        result = []
        for i in range(len(self.guesses)):
            result.append({
                'kamp':
                    self.post.params.basis.to_synth(self.post.params)
                    ['k' + str(i + 1)].value,
                'period': self.post.params['per' + str(i + 1)].value,
                'e': self.post.params.basis.to_synth(self.post.params)
                ['e' + str(i + 1)].value
            })
        return result

    def subtract_fit_from_data(self):
        """
        This method subtracts the model fit from the radial velocity data and
        returns the subtracted data.

        Returns:
            DataFrame: A DataFrame containing the fit subtracted data.
        """
        data_sub = deepcopy(self.data)
        data_sub['mnvel'] = (
                data_sub['mnvel']
                - self.post.likelihood.model(data_sub['time'].to_numpy())
        )
        return data_sub

    def mcmc_fit(self, **kwargs):
        """
        Performs MCMC fitting on the model.

        This method uses the radvel.mcmc function to perform MCMC fitting on
        the model. The results are stored in the
        'df' and 'df_synth' attributes of the object.

        Args:
            **kwargs: Arbitrary keyword arguments to be passed to the
            radvel.mcmc function.

        Returns:
            None
        """
        self.df = radvel.mcmc(self.post, savename='rawchains.h5', **kwargs)
        self.df_synth = self.post.params.basis.to_synth(self.df)

