import numpy as np
import emcee
from scipy.optimize import minimize
from pymultinest.solve import solve

def calc_normal_log_lik_vectorised(X,N,dataset):
    '''
    Calculate univariate normal log likelihood in a vectorised manner.
    This ignores all the offset value: -0.5*np.log(2*np.pi)
    :param data:
    :param X
    :param N
    :param dataset
    :return:
    '''

    mean = X[-2]
    std = X[-1]

    loglik = -0.5 * (N*np.log(std**2) + ((dataset -mean)**2 / std**2).sum())

    return loglik


if __name__ == '__main__':

    mean = 1
    std = 1
    N= int(1e4)
    np.random.seed(1)
    dataset = np.random.normal(mean, std, N)

    mean_est = 0.5
    std_est = 2
    log_lik = calc_normal_log_lik_vectorised(np.array([mean_est, std_est]),N, dataset)
    print(log_lik)

    # scipy optimiser
    print('########## Running scipy optimiser: ##########')
    X0 = np.array([mean_est, std_est])
    res = minimize(lambda x: -1*calc_normal_log_lik_vectorised(x,N,dataset), X0, method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': True})

    print(res.x)

    # multinest solver
    print('########## Running multinest: ##########')
    def normal_log_lik(X):
        '''
        Calculate univariate normal log likelihood in a vectorised manner.
        This ignores all the offset value: -0.5*np.log(2*np.pi)
        :param data:
        :param X
        :param N
        :param dataset
        :return:
        '''

        mean = X[-2]
        std = X[-1]
        N = len(dataset)
        loglik = -0.5 * (N * np.log(std ** 2) + ((dataset - mean) ** 2 / std ** 2).sum())

        return loglik

    def uniform_prior(X):
        X[0] = 10 * X[0] # uniform prior between 10 * [0,1]
        X[1] = 10 * X[1]  # uniform prior between 10* [0,1]
        return X

    parameters = ['mean', 'std_dev']
    result = solve(LogLikelihood=normal_log_lik, Prior=uniform_prior,
                   n_dims=len(X0), verbose=True)
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print('########## Running emcee: ##########')
    # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    def log_prior(X):
        # uninformative prior
        # https://emcee.readthedocs.io/en/stable/tutorials/line/
        mean = X[-2]
        std = X[-1]
        if std<0: # without this clause which excludes negative values for std, emcee cannot find the correct std value
            return -np.inf
        return 0

    def log_probability(X,N,dataset):
        lp = log_prior(X)
        if not np.isfinite(lp):
            return -np.inf
        return lp + calc_normal_log_lik_vectorised(X,N,dataset)


    nwalkers = 32
    ndim = len(X0)
    p0 = np.random.rand(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    log_probability,
                                    args=[N, dataset])
    state = sampler.run_mcmc(p0, 1000) # burn-in
    sampler.reset()
    sampler.run_mcmc(state, 1000, progress=True)  # actual run
    samples = sampler.get_chain(flat=True)
    print('parameter values:')
    for i, p in enumerate(parameters):
        print('%15s : %.3f +- %.3f' % (p, samples[:,i].mean(), samples[:,i].std()))