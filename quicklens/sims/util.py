# quicklens/sims/util.py
# --
# helper routines for use with the simulation libraries.
#

import numpy as np
import matplotlib.pyplot as plt
from .. import spec

def hash_check(hash1, hash2, ignore=[], keychain=[]):
    """ compare two hash dictionaries, usually produced by the .hashdict() method of a library object. """
    keys1 = hash1.keys()
    keys2 = hash2.keys()

    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)

    for key in set(keys1).union(set(keys2)):
        v1 = hash1[key]
        v2 = hash2[key]

        def hashfail(msg=None):
            print "ERROR: HASHCHECK FAIL AT KEY = " + ':'.join(keychain + [key])
            if msg != None:
                print "   ", msg
            print "   ", "V1 = ", v1
            print "   ", "V2 = ", v2
            assert(0)

        if type(v1) != type(v2):
            hashfail('UNEQUAL TYPES')
        elif type(v2) == dict:
            hash_check( v1, v2, ignore=ignore, keychain=keychain + [key] )
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not( v1 == v2 ):
                hashfail('UNEQUAL VALUES')


class ml_stats():
    '''
    Gather spectrum statistics
      lbins,         numpy-array definining the bin (lower-edge)
      t,             weight lambda-function for each ell bin (as applied in spec.lcl.get_ml())
      wb,            weight for each lbin
      bcl_av_sum,    Sum of binned Cl spectrum
      bcl_sq_sum,    Sum of binned Cl^2 spectrum
      bcl_cv_sum,    Sum of covariance

    '''

    def __init__(self, lbins, t=lambda l: 1., wb=None, cov=False):
        if wb == None:
            wb = np.ones(len(lbins) - 1)

        self.lbins = lbins
        self.t = t
        self.wb = wb
        self.docov = cov

        self.nsims = 0
        self.bcl_av_sum = np.zeros(len(lbins) - 1)
        self.bcl_sq_sum = np.zeros(len(lbins) - 1)

        if cov == True:
            self.bcl_cv_sum = np.zeros((len(lbins) - 1, len(lbins) - 1))

    def add(self, obj):
        tml = obj.get_ml(self.lbins, t=self.t).cl.real * self.wb

        self.nsims += 1
        self.bcl_av_sum += tml
        self.bcl_sq_sum += tml**2

        if self.docov == True:
            self.bcl_cv_sum += np.outer(tml, tml)

    def add_cl(self, obj):
        tml = obj.get_cl(self.lbins, t=self.t).cl.real * self.wb

        self.nsims += 1
        self.bcl_av_sum += tml
        self.bcl_sq_sum += tml**2

        if self.docov == True:
            self.bcl_cv_sum += np.outer(tml, tml)

    def avg(self):
        return spec.bcl(self.lbins, {'cl': self.bcl_av_sum / self.nsims})

    def std(self):
        return spec.bcl(self.lbins, {'cl': np.sqrt((self.bcl_sq_sum - self.bcl_av_sum**2 / self.nsims) / self.nsims)})

    def var(self):
        return spec.bcl(self.lbins, {'cl': (self.bcl_sq_sum - self.bcl_av_sum**2 / self.nsims) / self.nsims})

    def cov(self):
        assert(self.docov == True)
        return (self.bcl_cv_sum - np.outer(self.bcl_av_sum, self.bcl_av_sum) / self.nsims) / self.nsims

    def save(self, fname):
        '''
        Turn this object into a dictionary so it can be Pickled.
        Note: lambda functions can't be pickled, so you have to pass in the same lambda function on read-in.
        '''
        ml_stats_dict = {'lbins' : self.lbins,
                         'wb' : self.wb,
                         'docov' : self.docov,
                         'nsims' : self.nsims,
                         'bcl_av_sum' : self.bcl_av_sum,
                         'bcl_sq_sum' : self.bcl_sq_sum}
        if hasattr(self, 'bcl_cv_sum'): ml_stats_dict['bcl_cv_sum'] = self.bcl_cv_sum

        if not os.path.exists(fname):
            pk.dump(ml_stats_dict, open(fname,'w'))
        else:
            raise IOError("File already exists!")

    def load(self, fname):
        '''
        Load an object from a saved pickle file.
        Note: you must pass in the same lambda function for w as was used for the original object
        '''
        ml_stats_dict    = pk.load(open(fname,'r'))
        self.lbins       = ml_stats_dict['lbins']
        self.wb          = ml_stats_dict['wb']
        self.docov       = ml_stats_dict['docov']
        self.nsims       = ml_stats_dict['nsims']
        self.bcl_av_sum  = ml_stats_dict['bcl_av_sum']
        self.bcl_sq_sum  = ml_stats_dict['bcl_sq_sum']
        if ('bcl_cv_sum' in ml_stats_dict.keys()): self.bcl_cv_sum = ml_stats_dict['bcl_cv_sum']

    def plot_fill_between(self, t=lambda l: 1.0, m=None, **kwargs):
        '''
        plot this spectrum as a histogram
        t,            l-dependent coefficient.  e.g., t = lambda l : (l*(l+1.))**2 / (2.*np.pi) * 1.e7
        m,            the spectrum to be plotted
        '''
        ls = 0.5 * (self.lbins[0:-1] + self.lbins[1:])  # bin centers
        if m == None:
            m = self.avg().cl

        plt.fill_between(ls, t(ls) * (m - self.std().cl), t(ls) * (m + self.std().cl), **kwargs)

    def plot_error_bars(self, t=lambda l: 1.0, m=None, p=plt.errorbar, **kwargs):
        ls = 0.5 * (self.lbins[0:-1] + self.lbins[1:])  # bin centers
        if m == None:
            m = self.avg().cl

        p(ls, t(ls) * m, yerr=(t(ls) * self.std().cl), **kwargs)

class ml_cross_stats():
    '''
    Class to gater stats from a cross-spectrum.
    This emulates sims.util.ml_stats()
    '''
    def __init__(self, lbins, t=lambda l: 1., wb=None, cov=False):
        if wb == None:
            wb = np.ones(len(lbins) - 1)

        self.lbins = lbins
        self.t = t
        self.wb = wb
        self.docov = cov

        self.nsims = 0
        self.bcl_av_sum = np.zeros(len(lbins) - 1)
        self.bcl_sq_sum = np.zeros(len(lbins) - 1)

        if cov == True:
            self.bcl_cv_sum = np.zeros((len(lbins) - 1, len(lbins) - 1))

    def add(self, obj1, obj2):
        tml = spec.rcfft2cl(self.lbins, obj1, obj2, t=self.t).cl.real * self.wb

        self.nsims += 1
        self.bcl_av_sum += tml
        self.bcl_sq_sum += tml**2

        if self.docov == True:
            self.bcl_cv_sum += np.outer(tml, tml)

    def add_cl(self, obj1, obj2):
        tml = spec.rcfft2cl(self.lbins, obj1, obj2, t=self.t).cl.real * self.wb

        self.nsims += 1
        self.bcl_av_sum += tml
        self.bcl_sq_sum += tml**2

        if self.docov == True:
            self.bcl_cv_sum += np.outer(tml, tml)

    def avg(self):
        return spec.bcl(self.lbins, {'cl': self.bcl_av_sum / self.nsims})

    def std(self):
        return spec.bcl(self.lbins, {'cl': np.sqrt((self.bcl_sq_sum - self.bcl_av_sum**2 / self.nsims) / self.nsims)})

    def var(self):
        return spec.bcl(self.lbins, {'cl': (self.bcl_sq_sum - self.bcl_av_sum**2 / self.nsims) / self.nsims})

    def save(self, fname):
        '''
        Turn this object into a dictionary so it can be Pickled.
        Note: lambda functions can't be pickled, so you have to pass in the same lambda function on read-in.
        '''
        ml_stats_dict = {'lbins' : self.lbins,
                         'wb' : self.wb,
                         'docov' : self.docov,
                         'nsims' : self.nsims,
                         'bcl_av_sum' : self.bcl_av_sum,
                         'bcl_sq_sum' : self.bcl_sq_sum}
        if hasattr(self, 'bcl_cv_sum'): ml_stats_dict['bcl_cv_sum'] = self.bcl_cv_sum

        if not os.path.exists(fname):
            pk.dump(ml_stats_dict, open(fname,'w'))
        else:
            raise IOError("File already exists!")

    def load(self, fname):
        '''
        Load an object from a saved pickle file.
        Note: you must pass in the same lambda function for w as was used for the original object
        '''
        ml_stats_dict    = pk.load(open(fname,'r'))
        self.lbins       = ml_stats_dict['lbins']
        self.wb          = ml_stats_dict['wb']
        self.docov       = ml_stats_dict['docov']
        self.nsims       = ml_stats_dict['nsims']
        self.bcl_av_sum  = ml_stats_dict['bcl_av_sum']
        self.bcl_sq_sum  = ml_stats_dict['bcl_sq_sum']
        if ('bcl_cv_sum' in ml_stats_dict.keys()): self.bcl_cv_sum = ml_stats_dict['bcl_cv_sum']

    def plot_fill_between(self, t=lambda l: 1.0, m=None, **kwargs):
        '''
        plot this spectrum as a histogram
        t,            l-dependent coefficient.  e.g., t = lambda l : (l*(l+1.))**2 / (2.*np.pi) * 1.e7
        m,            the spectrum to be plotted
        '''
        ls = 0.5 * (self.lbins[0:-1] + self.lbins[1:])  # bin centers
        if m == None:
            m = self.avg().cl

        plt.fill_between(ls, t(ls) * (m - self.std().cl), t(ls) * (m + self.std().cl), **kwargs)

    def plot_error_bars(self, t=lambda l: 1.0, m=None, p=plt.errorbar, **kwargs):
        ls = 0.5 * (self.lbins[0:-1] + self.lbins[1:])  # bin centers
        if m == None:
            m = self.avg().cl

        p(ls, t(ls) * m, yerr=(t(ls) * self.std().cl), **kwargs)

