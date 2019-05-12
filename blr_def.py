from __future__ import division
import numpy as np
import tensorflow as tf

'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''

class BayesLR:
    _defl_var_scope_pfx = 'BayesLR_default_variable_scope_'
    _defl_var_scope_num = 0

    def __init__(self, featsize, M, a0=1., b0=1e2, var_scope=None, reuse=None, fltype=tf.float64, intype=tf.int32):
        '''
        * b0 is the scale; param in tf.random_gamma is inverse-scale
        * Y \in {0, 1}
        '''
        if var_scope is None:
            var_scope = BayesLR._defl_var_scope_pfx + str(BayesLR._defl_var_scope_num)
            BayesLR._defl_var_scope_num += 1
        self.a0 = a0; self.b0 = b0; self.var_scope = var_scope

        with tf.variable_scope(var_scope, reuse=reuse):
            theta = tf.get_variable('theta', initializer=tf.log(tf.random_gamma(shape=[M,1], alpha=a0, beta=1./b0, dtype=fltype)))
            w = tf.get_variable('w', initializer=tf.multiply( tf.random_normal(shape=[M, featsize], dtype=fltype), tf.exp(-.5*theta.initialized_value()) ))
        self.latvar = [w, theta]

        self.X_train = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_train = tf.placeholder(dtype=intype, shape=[None])

        X_test = tf.placeholder(dtype=fltype, shape=[None, featsize])
        Y_test = tf.placeholder(dtype=intype, shape=[None])
        inprod = tf.tensordot(w, X_test, [[1],[1]])
        probs = tf.sigmoid(inprod)
        prob = tf.reduce_mean(probs, axis=0)
        pred = tf.cast(tf.greater(prob, .5), intype)
        acc = tf.reduce_mean(tf.cast( tf.equal(pred, Y_test), fltype ))
        llh = tf.reduce_mean(tf.log( 1. - prob + (2.*prob - 1.) * tf.cast(Y_test, fltype) ))
        self.X_test = X_test; self.Y_test = Y_test; self.prob = prob; self.pred = pred; self.acc = acc; self.llh = llh

    def get_logp(self, w, theta, fullsize):
        theta = tf.squeeze(theta)
        alpha = tf.exp(theta)
        inprod = tf.tensordot(w, self.X_train, [[1],[1]])
        logp = (self.a0 + .5 * w.get_shape().as_list()[1]) * theta - alpha / self.b0 \
                - .5 * alpha * tf.reduce_sum(w*w, axis=1) \
                + fullsize * tf.reduce_mean( inprod * tf.cast(self.Y_train, w.dtype) - tf.log(1. + tf.exp(inprod)), axis=1 )
        return logp

