from __future__ import division

import numpy as np
from collections import deque, Counter
from lda_sample_z_ids import sample_z_ids
from scipy.misc import logsumexp

class LDA(object):
    def __init__(self, D, W, K, alpha, beta, sigma, n_gsamp):
        self.D = D
        self.W = W
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.n_gsamp = n_gsamp

        self.epoch = 0
        self.nIter = 0
        self.tr_epochs = []
        self.tr_nIters = []
        self.tr_logperps = []
        self.ho_epochs = []
        self.ho_nIters = []
        self.ho_logperps = []

    #---------------------#

    def _sample_gibbs(self, phi, train_cts):
        batch_D = len(train_cts)
        batch_N = sum(sum(ddict.values()) for ddict in train_cts)
        M = phi.shape[0]
        uni_rvs = np.random.uniform(size = (M, batch_N*(self.n_gsamp+1)))
        Adk_mean = np.zeros((M, batch_D, self.K))
        Bkw_mean = np.zeros((M, self.K, self.W))
        burn_in = self.n_gsamp // 2
        sample_z_ids(Adk_mean, Bkw_mean,
                phi, uni_rvs, train_cts, self.alpha, self.n_gsamp, burn_in)
        return (Adk_mean, Bkw_mean)

    def get_grad_logp(self, tr_train_cts, theta):
        if theta.ndim == 2: theta = theta[None,:,:]
        batch_D = len(tr_train_cts)
        # robust softmax:
        phi = np.exp(theta - logsumexp(theta, axis=-1, keepdims=True)) # theta, phi: [K, W]
        Adk_mean, Bkw_mean = self._sample_gibbs(phi, tr_train_cts)
        grad = (self.beta - theta) / (self.sigma**2) \
                + (self.D/batch_D) * (Bkw_mean - phi * Bkw_mean.sum(axis=-1, keepdims=True))
                # + (self.D/batch_D) * (Bkw_mean - phi * np.expand_dims(Adk_mean.sum(axis=-2), axis=-1))
        self.epoch += batch_D / self.D
        self.nIter += 1
        return grad#, phi

    #---------------------#

    def _logperp(self, phi, train_cts, test_cts, test_nwords=None):
        Adk_mean, Bkw_mean = self._sample_gibbs(phi, train_cts)
        eta_hat = Adk_mean + self.alpha
        eta_hat /= eta_hat.sum(axis=-1, keepdims=True)
        M = phi.shape[0]
        sum_logperp = sum(cntr[w] * np.log(np.sum(eta_hat[:,d,:] * phi[:,:,w]) / M)
                for (d, cntr) in enumerate(test_cts) for w in cntr)
        return - sum_logperp / ( sum(sum(cntr.values()) for cntr in test_cts) if test_nwords is None else test_nwords )

    def get_training_logperp(self, tr_train_cts, tr_test_cts, theta=None, phi=None):
        if phi is None: # robust softmax:
            phi = np.exp(theta - logsumexp(theta, axis=-1, keepdims=True)) # theta, phi: [K, W]
        if phi.ndim == 2: phi = phi[None,:,:]
        res = self._logperp(phi, tr_train_cts, tr_test_cts)
        self.tr_epochs.append(self.epoch)
        self.tr_nIters.append(self.nIter)
        self.tr_logperps.append(res)
        return res, phi

    def set_holdout_logperp(self, perpType, ho_train_cts, ho_test_cts, n_window=None):
        self.perpType = perpType
        self.ho_train_cts = ho_train_cts
        self.ho_test_cts = ho_test_cts
        self.ho_test_nwords = sum(sum(cntr.values()) for cntr in ho_test_cts)
        self.n_eval = 0
        if perpType == 'para': pass
        elif perpType == 'seq':
            self.ho_avg_probs = {(d, w): 0.0 for (d, cntr) in enumerate(ho_test_cts) for w in cntr}
        elif perpType == 'window':
            if type(n_window) is not int: raise TypeError('"n_window" has to be provided and of type int!')
            self.n_window = n_window
            self.ho_sum_probs_dq = {(d, w): [0.0, deque()] for (d, cntr) in enumerate(ho_test_cts) for w in cntr}
        else: raise ValueError('Unknown "perpType" {}!'.format(perpType))

    def get_holdout_logperp(self, theta=None, phi=None):
        self.n_eval += 1
        if phi is None: # robust softmax:
            phi = np.exp(theta - logsumexp(theta, axis=-1, keepdims=True)) # theta, phi: [K, W]
        if phi.ndim == 2: phi = phi[None,:,:]

        if self.perpType == 'para':
            res = self._logperp(phi, self.ho_train_cts, self.ho_test_cts, self.ho_test_nwords)
        else:
            if phi.ndim != 3 or phi.shape[0] != 1: raise ValueError('"phi" has to be of shape [1,:,:] for mode "seq" or "window"!')
            Adk_mean, Bkw_mean = self._sample_gibbs(phi, self.ho_train_cts)
            eta_hat = Adk_mean + self.alpha
            eta_hat /= eta_hat.sum(axis=-1, keepdims=True)
            if self.perpType == 'seq':
                self.ho_avg_probs = {(d, w): (1-1./self.n_eval) * self.ho_avg_probs[(d, w)] \
                                           + (1./self.n_eval) * np.dot(eta_hat[0, d, :], phi[0, :, w])
                             for (d, w) in self.ho_avg_probs}
                sum_logperp = sum(cntr[w] * np.log(self.ho_avg_probs[(d, w)])
                                 for (d, cntr) in enumerate(self.ho_test_cts) for w in cntr)
                res = - sum_logperp / self.ho_test_nwords
            elif self.perpType == 'window':
                sum_logperp = 0
                for (d, w), p in self.ho_sum_probs_dq.items():
                    val = np.dot(eta_hat[0, d, :], phi[0, :, w])
                    p[0] += val; p[1].append(val)
                    if self.n_eval > self.n_window: p[0] -= p[1].popleft()
                    sum_logperp += self.ho_test_cts[d][w] * np.log(p[0] / min(self.n_eval, self.n_window))
                res = - sum_logperp / self.ho_test_nwords
            else: raise ValueError('Unknown "perpType" {}!'.format(self.perpType))
        self.ho_epochs.append(self.epoch)
        self.ho_nIters.append(self.nIter)
        self.ho_logperps.append(res)
        return res#, phi

    #---------------------#

    def save_dict(self):
        return {attr: getattr(self, attr) for attr in vars(self)
                if attr not in {'ho_train_cts', 'ho_test_cts', 'ho_test_nwords', 'ho_avg_probs', 'ho_sum_probs_dq'}}

#---------------------#

class Dataset(object):
    def __init__(self, filename, batchsize, train_ratio=.8):
        '''
        data file format: n_docs \n n_docs * [word1id, word2id, ...]
        data object format: [{wordid_di: count}] for d over documents and i over words in d.
        '''
        self.tr_train_cts = []
        self.tr_test_cts = []
        self.ho_train_cts = []
        self.ho_test_cts = []
        self.batchsize = batchsize
        self._nIter = 0
        with open(filename, 'r') as fid:
            self.n_docs = int(fid.readline())
            self.n_tr = int(round(self.n_docs * train_ratio))
            self.n_ho = self.n_docs - self.n_tr
            ho_idx = np.random.choice(self.n_docs, self.n_ho, replace=False)
            for d in range(self.n_docs):
                words_d = [int(num) for num in fid.readline().split()]
                if len(words_d) < 2: raise ValueError('too few words in line {:d}!'.format(d+2))
                train_cts_d = dict(Counter([num for (j, num) in enumerate(words_d) if j%10 != 0]))
                test_cts_d = dict(Counter([num for (j, num) in enumerate(words_d) if j%10 == 0]))
                if d in ho_idx:
                    self.ho_train_cts.append(train_cts_d)
                    self.ho_test_cts.append(test_cts_d)
                else:
                    self.tr_train_cts.append(train_cts_d)
                    self.tr_test_cts.append(test_cts_d)

    def get_batch(self, nIter=None):
        if nIter is not None: self._nIter = nIter
        batch = [i % self.n_tr for i in range(self._nIter * self.batchsize, (self._nIter+1) * self.batchsize)]
        self._nIter += 1
        return [self.tr_train_cts[i] for i in batch], [self.tr_test_cts[i] for i in batch]

