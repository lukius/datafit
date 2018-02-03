import scipy

import numpy as np
import matplotlib.pyplot as plt


class DataPlotter(object):

    # Given a data set and a sequence of distribution names, this object
    # generates histogram charts to visualize the data, adjusts each
    # distribution to these data and plots the corresponding curves.
    # It can be used as a complement to the DataClassifier object in order to
    # visually validate its output.

    def __init__(self, data):
        self.data = data

    def plot(self, distributions=None, title=None, filename=None, bins=None):
        x = np.linspace(min(self.data)-1, max(self.data)+1)
        bins = bins or 50
        distributions = distributions or list()
        for distribution in distributions:
            dist = getattr(scipy.stats, distribution)
            params = dist.fit(self.data)
            fitted_pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
            plt.plot(x, fitted_pdf, label=distribution)
        plt.xlim(min(self.data)-1, max(self.data)+1 )
        plt.hist(self.data, bins=bins, normed=1, alpha=0.3)
        plt.legend(loc='upper right')
        if title is not None:
            plt.title(title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def plot_with_params(self, distribution, params, title=None, filename=None, bins=None):
        x = np.linspace(min(self.data)-1, max(self.data)+1)
        bins = bins or 50
        dist = getattr(scipy.stats, distribution)

        pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
        plt.plot(x, pdf, label=distribution)

        real_params = dist.fit(self.data)
        fitted_pdf = dist.pdf(x, *real_params[:-2], loc=real_params[-2], scale=real_params[-1])
        plt.plot(x, fitted_pdf, label='%s (fitted)' % distribution)

        plt.xlim(min(self.data)-1, max(self.data)+1 )
        plt.hist(self.data, bins=bins, normed=1, alpha=0.3)
        plt.legend(loc='upper right')
        if title is not None:
            plt.title(title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
        plt.close()
