import tensorflow as tf
import numpy as np

'''
    Sample code to reproduce our results for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15) https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.
    
    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)
    
    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda) 
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.
    
    Copyright (c) 2016,  Qiang Liu & Dilin Wang
    All rights reserved.
'''

class BayesNN:
    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batchsize: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    _varscope_pfx = 'BayesNN_default_variable_scope_'
    _varscope_num = 0
    def __init__(self, featsize, M, n_hidden=50, a0=1., b0=10., var_scope=None, reuse=None, fltype=tf.float64, Y_std=1.):
        # b0 is the scale; param in tf.random_gamma is inverse-scale
        if var_scope is None:
            var_scope = BayesNN._varscope_pfx + str(BayesNN._varscope_num)
            BayesNN._varscope_num += 1
        self.featsize = featsize
        self.a0 = a0; self.b0 = b0; self.var_scope = var_scope
        self.num_vars = self.featsize * n_hidden + n_hidden * 2 + 3  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances

        self.X_train = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_train = tf.placeholder(dtype=fltype, shape=[None])
        with tf.variable_scope(var_scope, reuse=reuse):
            loggamma = tf.get_variable('loggamma', initializer=tf.log(tf.random_gamma(shape=[M, 1], alpha=a0, beta=1./b0, dtype=fltype)))
            loglambda = tf.get_variable('loglambda', initializer=tf.log(tf.random_gamma(shape=[M, 1], alpha=a0, beta=1./b0, dtype=fltype)))
            ####
            # w1 = tf.get_variable('w1', initializer=tf.multiply( tf.random_normal(shape=[M, n_hidden, featsize], dtype=fltype),
            #     tf.expand_dims(tf.exp(-.5*loglambda.initialized_value()), 2) ))
            # b1 = tf.get_variable('b1', initializer=tf.multiply( tf.random_normal(shape=[M, n_hidden], dtype=fltype), tf.exp(-.5*loglambda.initialized_value()) ))
            # w2 = tf.get_variable('w2', initializer=tf.multiply( tf.random_normal(shape=[M, n_hidden], dtype=fltype), tf.exp(-.5*loglambda.initialized_value()) ))
            # b2 = tf.get_variable('b2', initializer=tf.multiply( tf.random_normal(shape=[M, 1], dtype=fltype), tf.exp(-.5*loglambda.initialized_value()) ))
            ####
            w1 = tf.get_variable('w1', shape=[M, n_hidden, featsize], dtype=fltype, initializer=tf.random_normal_initializer(stddev=1./np.sqrt(featsize+1.)))
            b1 = tf.get_variable('b1', shape=[M, n_hidden], dtype=fltype, initializer=tf.zeros_initializer())
            w2 = tf.get_variable('w2', shape=[M, n_hidden], dtype=fltype, initializer=tf.random_normal_initializer(stddev=1./np.sqrt(n_hidden+1.)))
            b2 = tf.get_variable('b2', shape=[M, 1], dtype=fltype, initializer=tf.zeros_initializer())
            y_mean = self._get_y_mean(w1, b1, w2, b2, self.X_train) # X_train is fed with Unnormalized X
            self.init_loggamma = loggamma.assign(-tf.log(tf.reduce_mean((y_mean - self.Y_train)**2, axis=1, keepdims=True))) # Y_train is fed with Unnormalized Y
            ####
        self.latvar = [w1, b1, w2, b2, loggamma, loglambda]

        self.X_test = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_test = tf.placeholder(dtype=fltype, shape=[None])
        y_preds = self._get_y_mean(w1, b1, w2, b2, self.X_test)
        self.y_pred = tf.reduce_mean(y_preds, axis=0)
        self.rmse = tf.sqrt(tf.reduce_mean((self.y_pred - self.Y_test)**2))

        self.X_dev = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_dev = tf.placeholder(dtype=fltype, shape=[None])
        y_mean_dev = self._get_y_mean(w1, b1, w2, b2, self.X_dev)
        gamma0 = tf.exp(loggamma)
        invgamma1 = (Y_std**2) * tf.reduce_mean((y_mean_dev - self.Y_dev)**2, axis=1, keepdims=True) # dev llh is for Unnormalized Y
        gamma1 = 1./invgamma1
        lik0 = .5*loggamma - .5*gamma0 * invgamma1 # dev llh is for Unnormalized Y
        lik1 = .5*tf.log(gamma1) - .5 # dev llh is for Unnormalized Y
        gamma = tf.where(tf.greater(lik0, lik1), gamma0, gamma1)
        probs = tf.sqrt(gamma) / np.sqrt(2*np.pi) * tf.exp(-.5*gamma*(Y_std**2) * (y_preds - self.Y_test)**2) # test llh is for Unnormalized Y
        self.llh = tf.reduce_mean(tf.log(tf.reduce_mean(probs, axis=0)))

    def _get_y_mean(self, w1, b1, w2, b2, X):
        return tf.reduce_sum(tf.nn.relu( tf.tensordot(w1, X, axes=[[2],[1]]) + tf.expand_dims(b1, 2) ) * tf.expand_dims(w2, 2), axis=1) + b2

    def get_logp(self, w1, b1, w2, b2, loggamma, loglambda, fullsize):
        y_mean = self._get_y_mean(w1, b1, w2, b2, self.X_train)
        b2 = tf.squeeze(b2)
        loggamma = tf.squeeze(loggamma)
        loglambda = tf.squeeze(loglambda)
        mean_log_lik_data = .5 * (loggamma - np.log(2*np.pi)) - .5*tf.exp(loggamma) * tf.reduce_mean((y_mean-self.Y_train)**2, axis=1) # train llh is for Normalized Y
        log_prior_data = self.a0*loggamma - tf.exp(loggamma)/self.b0
        log_prior_w = .5 * (self.num_vars-2) * (loglambda - np.log(2*np.pi)) - .5*tf.exp(loglambda) * (\
                tf.reduce_sum(w1**2, axis=[1,2]) + tf.reduce_sum(w2**2, axis=1) + tf.reduce_sum(b1**2, axis=1) + b2**2\
            ) + self.a0*loglambda - tf.exp(loglambda)/self.b0
        return fullsize * mean_log_lik_data + log_prior_data + log_prior_w

class Dataset:
    def __init__(self, filename, batchsize, train_ratio=.9, dev_ratio=.1):
        self.batchsize = batchsize
        self._data = np.loadtxt(filename)
        self.featsize = self._data.shape[1] - 1
        self.allsize = self._data.shape[0]
        traindevsize = int(round(self.allsize * train_ratio))
        self.testsize = self.allsize - traindevsize
        self.devsize = min(int(round(traindevsize * dev_ratio)), 500)
        self.trainsize = traindevsize - self.devsize
        self._nIter = 0
        self._split_and_normalize()

    def reset(self):
        self._data[:, :-1] *= self.X_std
        self._data[:, :-1] += self.X_mean
        self._data[:, -1] *= self.Y_std
        self._data[:, -1] += self.Y_mean
        self._nIter = 0
        self._split_and_normalize()

    def _split_and_normalize(self):
        ridx = np.random.choice(self.allsize, self.allsize, replace=False)
        X_train = self._data[:ridx[self.trainsize], :-1]
        Y_train = self._data[:ridx[self.trainsize], -1]
        self.X_mean = np.mean(X_train, 0)
        self.X_std = np.std(X_train, 0)
        self.X_std[self.X_std == 0] = 1
        self.Y_mean = np.mean(Y_train)
        self.Y_std = np.std(Y_train)
        if np.any(self.Y_std == 0): raise ValueError('unexpected Y_std == 0')
        self._data[:, :-1] -= self.X_mean
        self._data[:, :-1] /= self.X_std
        self._data[:, -1] -= self.Y_mean
        self._data[:, -1] /= self.Y_std
        self.X_train = self._data[ridx[:self.trainsize], :-1]
        self.Y_train = self._data[ridx[:self.trainsize], -1]
        self.X_dev = self._data[ridx[self.trainsize: self.trainsize+self.devsize], :-1]
        self.Y_dev = self._data[ridx[self.trainsize: self.trainsize+self.devsize], -1]
        self.X_test = self._data[ridx[-self.testsize:], :-1]
        self.Y_test = self._data[ridx[-self.testsize:], -1]

    def get_batch(self, nIter=None):
        if nIter is not None: self._nIter = nIter
        batch = [i % self.trainsize for i in range(self._nIter * self.batchsize, (self._nIter+1) * self.batchsize)]
        self._nIter += 1
        return self.X_train[batch, :], self.Y_train[batch]

    def get_batch_for_init_loggamma(self):
        ridx = np.random.choice(self.trainsize, np.min([self.trainsize, 1000]), replace=False)
        return self.X_train[ridx, :], self.Y_train[ridx]

