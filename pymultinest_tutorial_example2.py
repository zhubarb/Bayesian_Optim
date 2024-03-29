# https://github.com/JohannesBuchner/pymultinest-tutorial/blob/master/example2/lines.py
import numpy as np
import scipy.stats, scipy
import pymultinest
import json
import sys
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
import matplotlib.pyplot as plt

x = numpy.linspace(0, 1, 400)

noise = 0.1

# model for 2 gaussians, same width, fixed offset
def model(pos1, width, height1, height2):
	pos2 = pos1 + 0.05
	return  height1 * scipy.stats.norm.pdf(x, pos1, width) + \
		height2 * scipy.stats.norm.pdf(x, pos2, width)

def generate_data() ->np.ndarray:
    numpy.random.seed(5)
    for i in range(8):
        pos1 = numpy.random.uniform(0.1, 0.75)
        height1 = 10**numpy.random.uniform(-2.5, -2)
        width = (0.0001 / height1) + 0.0001
        height2 = height1 * pos1
        print('%02d %.2f %.4f %.4f %.4f' % (i+1, pos1, width, height1, height2))
        ymodel = model(pos1, width, height1, height2)
        ydata = numpy.random.normal(ymodel, noise)
        numpy.savetxt("out/spectrum%02d" % (i+1), ydata)
        plt.plot(x, ymodel, '-', color='blue', label='model: %f %f %f %f' % (pos1, width, height1, height2))
        plt.plot(x, ydata, '+ ', color='red', label='data')
        plt.savefig("out/spectrum%02d.eps" % (i+1))
        plt.close()


x = numpy.linspace(0, 1, 400)
ydata = None  # loaded below
noise = 0.1


# model for 2 gaussians, same width, fixed offset
def model(pos1, width, height1, height2):
    pos2 = pos1 + 0.05
    return height1 * scipy.stats.norm.pdf(x, pos1, width) + \
        height2 * scipy.stats.norm.pdf(x, pos2, width)


# a more elaborate prior
# parameters are pos1, width, height1, [height2]
def prior(cube, ndim, nparams):
    # cube[0] = cube[0]            # uniform prior between 0:1
    cube[1] = 10 ** (cube[1] * 8 - 4)  # log-uniform prior between 10^-4 and 10^4
    cube[2] = 10 ** (cube[2] * 4 - 4)  # log-uniform prior between 10^-4 and 1
    if ndim < 4:
        return
    cube[3] = 10 ** (cube[3] * 4 - 4)  # log-uniform prior between 10^-4 and 1


def loglike(cube, ndim, nparams):
    pos1, width, height1 = cube[0], cube[1], cube[2]
    height2 = cube[3] if ndim > 3 else 0
    ymodel = model(pos1, width, height1, height2)
    loglikelihood = (-0.5 * ((ymodel - ydata) / noise) ** 2).sum()
    return loglikelihood


if __name__ == '__main__':
    generate_data()
    # analyse the file given as first argument
    datafile = './out/spectrum01'
    ydata = numpy.loadtxt(datafile)

    # analyse with 1 gaussian

    # number of dimensions our problem has
    parameters = ["pos1", "width", "height1"]
    n_params = len(parameters)

    # run MultiNest
    pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile + '_1_', resume=False, verbose=True)
    json.dump(parameters, open(datafile + '_1_params.json', 'w'))  # save parameter names

    # plot the distribution of a posteriori possible models
    plt.figure()
    plt.plot(x, ydata, '+ ', color='red', label='data')
    a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params=n_params)
    for (pos1, width, height1) in a.get_equal_weighted_posterior()[::100, :-1]:
        plt.plot(x, model(pos1, width, height1, 0), '-', color='blue', alpha=0.3, label='data')

    plt.savefig(datafile + '_1_posterior.pdf')
    plt.close()

    a_lnZ = a.get_stats()['global evidence']
    print('************************')
    print('MAIN RESULT: Evidence Z ')
    print('************************')
    print('  log Z for model with 1 line = %.1f' % (a_lnZ / log(10)))
