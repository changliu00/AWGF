from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from six.moves import zip, map

def get_sinkhorn_lse(x, y, M, N, invReg, maxIter):
    # with tf.control_dependencies([tf.assert_equal(x.get_shape()[1:], y.get_shape()[1:])]):
    #     M = x.get_shape()[0]
    #     N = y.get_shape()[0]
    with tf.control_dependencies([tf.assert_equal(tf.shape(x)[0], M), tf.assert_equal(tf.shape(y)[0], N)]):
        fltype = x.dtype
        M_fl = tf.cast(M, fltype)
        N_fl = tf.cast(N, fltype)
        log_M = tf.log(M_fl)
        log_N = tf.log(N_fl)
        rg = tf.range(1, tf.rank(x))

        inprod_ij = tf.tensordot(x, y, axes=tf.stack([rg, rg]))
        norm2x_i = tf.reduce_sum(x*x, axis=rg)
        norm2y_i = tf.reduce_sum(y*y, axis=rg)
        dist2_ij = norm2y_i + tf.expand_dims(norm2x_i, 1) - 2 * inprod_ij
        log_K_ij = -invReg * dist2_ij
        def loop_body(i, log_u):
            log_u_new = log_N - log_M - tf.reduce_logsumexp(log_K_ij - tf.reduce_logsumexp(log_K_ij + tf.expand_dims(log_u, 1), 0), 1)
            log_u_new.set_shape([M])
            return [i+1, log_u_new]
        _, log_u_out = tf.while_loop(\
            lambda i, log_u: tf.less(i, maxIter),\
            loop_body,\
            [0, - log_M * tf.ones([M], dtype=fltype)],\
            back_prop=False)
        log_v = - log_N - tf.reduce_logsumexp(log_K_ij + tf.expand_dims(log_u_out, 1), 0)
        P_ij = tf.exp(log_K_ij + tf.expand_dims(log_u_out, 1) + log_v)
        wdist = tf.reduce_sum(dist2_ij * P_ij)
        vecfld = tf.tensordot(P_ij, y, [[1], [0]]) - x
        return wdist, P_ij, vecfld

def get_ipot_lse(x, y, M, N, invReg, maxIter):
    # with tf.control_dependencies([tf.assert_equal(x.get_shape()[1:], y.get_shape()[1:])]):
    #     M = x.get_shape()[0]
    #     N = y.get_shape()[0]
    with tf.control_dependencies([tf.assert_equal(tf.shape(x)[0], M), tf.assert_equal(tf.shape(y)[0], N)]):
        fltype = x.dtype
        M_fl = tf.cast(M, fltype)
        N_fl = tf.cast(N, fltype)
        log_M = tf.log(M_fl)
        log_N = tf.log(N_fl)
        rg = tf.range(1, tf.rank(x))

        inprod_ij = tf.tensordot(x, y, axes=tf.stack([rg, rg]))
        norm2x_i = tf.reduce_sum(x*x, axis=rg)
        norm2y_i = tf.reduce_sum(y*y, axis=rg)
        dist2_ij = norm2y_i + tf.expand_dims(norm2x_i, 1) - 2 * inprod_ij
        log_K_ij = -invReg * dist2_ij
        def loop_body(i, log_v, log_P_ij):
            log_Q_ij = log_K_ij + log_P_ij
            log_u_new = -log_M - tf.reduce_logsumexp(log_Q_ij + log_v, 1)
            log_v_new = -log_N - tf.reduce_logsumexp(log_Q_ij + tf.expand_dims(log_u_new,1), 0)
            log_P_ij_new = log_Q_ij + tf.expand_dims(log_u_new,1) + log_v_new
            log_v_new.set_shape([N])
            log_P_ij_new.set_shape([M,N])
            return [i+1, log_v_new, log_P_ij_new]
        _, log_v_out, log_P_ij_out = tf.while_loop(\
            lambda i, log_v, log_P_ij: tf.less(i, maxIter),\
            loop_body,\
            [0, - log_N * tf.ones([N], dtype=fltype), tf.zeros([M,N], dtype=fltype)],\
            back_prop=False)
        P_ij = tf.exp(log_P_ij_out)
        wdist = tf.reduce_sum(dist2_ij * P_ij)
        vecfld = tf.tensordot(P_ij, y, [[1], [0]]) - x
        return wdist, P_ij, vecfld

class DynamicsInfo():
    def __init__(self, L_samples, L_particles, L_grad_logp, L_vecfld, global_step, bandw, ibandw, bwmed, bwmax, var_scope):
        self.L_samples = L_samples
        self.L_particles = L_particles
        self.L_grad_logp = L_grad_logp
        self.L_vecfld = L_vecfld
        self.global_step = global_step
        self.bandw = bandw
        self.ibandw = ibandw
        self.bwmed = bwmed
        self.bwmax = bwmax
        self.var_scope = var_scope

class Dynamics():
    _defl_var_scope_pfx = 'Dynamics_default_variable_scope_'
    _defl_var_scope_num = 0

    def __init__(self, dnType, pm):
        self.dnType = dnType
        self.pm = pm
    
    def evolve(self, L_samples, get_logp = None, L_grad_logp = None, global_step = None, var_scope = None, reuse = None):
        if (get_logp is None) == (L_grad_logp is None): raise ValueError('Exactly one of "get_logp" or "L_grad_logp" should be passed.')
        if type(L_samples) != list: L_samples = [L_samples]
        M = L_samples[0].get_shape()[0]
        if not all([samples.get_shape()[0] == M for samples in L_samples]): raise ValueError('Sample sizes of all variables are not the same!')
        if var_scope is None:
            var_scope = Dynamics._defl_var_scope_pfx + str(Dynamics._defl_var_scope_num)
            Dynamics._defl_var_scope_num += 1
        if global_step is None:
            with tf.variable_scope(var_scope, reuse=reuse):
                global_step = tf.get_variable('global_step', initializer=tf.constant(0))
        intype = global_step.initialized_value().dtype
        fltype = L_samples[0].initialized_value().dtype
        global_step = global_step.assign_add(1)
        global_step_fl = tf.cast(global_step, fltype)
        D = tf.add_n([tf.reduce_prod(samples.get_shape()[1:]) for samples in L_samples])
        D_fl = tf.cast(D, fltype)
        M_fl = tf.cast(M, fltype)
        L_rk = [tf.rank(samples) for samples in L_samples]
        L_rg = [tf.range(1, rk) for rk in L_rk]

        # preparation
        def get_name(var):
            name = var.name
            return name[(name.rfind('/')+1) : name.rfind(':')]
        with tf.variable_scope(var_scope, reuse=reuse):
            # "samples" is the final samples we want to use in the model; "particles" is where the gradient is estimated.
            if self.pm.accType == 'wgd':
                L_particles = L_samples
            elif self.pm.accType == 'po':
                L_diffsamples = [tf.get_variable('diff_' + get_name(samples), shape=samples.initialized_value().get_shape(), dtype=fltype, initializer=tf.zeros_initializer()) for samples in L_samples]
                L_particles = L_samples
            elif self.pm.accType in {'wnag', 'wnag-sink', 'wnes0', 'wnes1', 'wnes1-sink'}:
                L_auxsamples = [tf.get_variable('aux_' + get_name(samples), initializer=samples.initialized_value()) for samples in L_samples]
                L_oldsamples = [tf.get_variable('old_' + get_name(samples), initializer=samples.initialized_value()) for samples in L_samples]
                L_particles = L_auxsamples
            else: raise ValueError('unknown "accType": "{}"!'.format(self.pm.accType))
            if get_logp is not None:
                logp = get_logp(*L_particles)
                L_grad_logp = tf.gradients(logp, L_particles, stop_gradients=L_particles)
            else:
                if type(L_grad_logp) != list: L_grad_logp = [L_grad_logp]

        # bandwidth
        inprod_ij = tf.add_n([tf.tensordot(particles, particles, axes=tf.stack([rg, rg])) for particles, rg in zip(L_particles, L_rg)])
        norm2_i = tf.diag_part(inprod_ij)
        dist2_ij = norm2_i + tf.reshape(norm2_i, [M,1]) - 2 * inprod_ij
        bwmed = .5 * tf.contrib.distributions.percentile(dist2_ij, q=50., interpolation='higher') / tf.log(M_fl+1.) # np version is 'midpoint', which tf does not support.
        bwmax = tf.reduce_max(dist2_ij) / D_fl

        if M == 1:
            ibandw = tf.constant(1., dtype=fltype)
        else:
            if not hasattr(self.pm, 'bwType') or self.pm.bwType == None:
                ibandw = tf.constant(1., dtype=fltype)
            elif self.pm.bwType == 'fix':
                ibandw = tf.constant(1./self.pm.bwVal, dtype=fltype)
            elif self.pm.bwType == 'med':
                ibandw = 1./bwmed
            elif self.pm.bwType == 'he':
                with tf.variable_scope(var_scope, reuse=reuse):
                    ibandw = tf.get_variable('ibandw', initializer=tf.constant(1., dtype=fltype))
                def _get_obj_ibandw(ibw):
                    exp_ij = tf.exp(-.5 * ibw * dist2_ij)
                    exp_i = tf.reduce_sum(exp_ij, axis=1)
                    nor_exp_ij = exp_ij / exp_i
                    expinprod_i = tf.reduce_sum(exp_ij * inprod_ij, axis=1)
                    bwobj_i = tf.pow(ibw, .5*D_fl) * \
                            (   (ibw ** 2) * \
                                (   tf.reduce_sum( dist2_ij * exp_ij, axis=1) \
                                    - expinprod_i + tf.tensordot( exp_ij, norm2_i, [[1],[0]] ) \
                                    + tf.reduce_sum( tf.matmul( inprod_ij, exp_ij ) * nor_exp_ij, axis=1 ) \
                                    - tf.tensordot( nor_exp_ij, expinprod_i, [[1],[0]] ) \
                                )
                                - ibw * D_fl * exp_i \
                            ) # [-D-2]
                    if hasattr(self.pm, 'bwSubType') or self.pm.bwSubType == 'a':
                        ## scaled L2(q) sq: [-2D+4]
                        return tf.reduce_sum(bwobj_i**2) / (ibw**4)
                    elif self.pm.bwSubType == 'b':
                        ## L2(q) sq: [-2D-4]
                        return tf.reduce_sum(bwobj_i**2)
                    elif self.pm.bwSubType == 'c':
                        ## L2(unif) sq: [-D-4]
                        return tf.reduce_sum(bwobj_i**2 / exp_i) / tf.pow(ibw, .5*D_fl)
                    elif self.pm.bwSubType == 'd':
                        ## L2(q^-1) sq: [-4]
                        return tf.reduce_sum((bwobj_i / exp_i)**2) / tf.pow(ibw, D_fl)
                    elif self.pm.bwSubType == 'e':
                        ## scaled L2(q) sq: [2D-4]
                        return tf.reduce_sum(bwobj_i**2) / tf.pow(ibw, 2*D_fl)
                    elif self.pm.bwSubType == 'f':
                        ## L1(unif): [-2]
                        return tf.reduce_sum(tf.abs(bwobj_i / exp_i)) / tf.pow(ibw, .5*D_fl)
                    elif self.pm.bwSubType == 'g':
                        ## normalized L2(unif) sq: [D-4]
                        return tf.reduce_sum((bwobj_i / exp_i)**2 / exp_i) / tf.pow(ibw, 1.5*D_fl)
                    elif self.pm.bwSubType == 'h':
                        ## scaled L2(q) sq: [0]
                        return tf.reduce_sum(bwobj_i**2) / tf.pow(ibw, D_fl+2)
                    else: raise ValueError('unknown "bwSubType": "{}"!'.format(self.pm.bwSubType))

                ibw_in = tf.cond(tf.equal(global_step, 1), lambda: 1./bwmed, lambda: ibandw.read_value())
                # ## (1) naive gradient descent
                # _, ibw_out = tf.while_loop(\
                #     lambda i, ibw: tf.less(i, self.pm.bwMaxIter),\
                #     lambda i, ibw: [i+1,\
                #         tf.abs( ibw - self.pm.bwStepsize * tf.squeeze(tf.gradients(_get_obj_ibandw(ibw), ibw, stop_gradients=ibw)) )],\
                #     [0, ibw_in], back_prop=False, parallel_iterations=1)
                ## (2) one-step line search
                explore_ratio = tf.constant(1.1, dtype=fltype)
                obj_ibw_in = _get_obj_ibandw(ibw_in)
                grad_ibw_in = tf.squeeze(tf.gradients(obj_ibw_in, ibw_in, stop_gradients=ibw_in))
                ibw_1 = tf.cond(tf.less(grad_ibw_in, 0.), lambda: ibw_in*explore_ratio, lambda: ibw_in/explore_ratio)
                obj_ibw_1 = _get_obj_ibandw(ibw_1)
                slope_ibw = (obj_ibw_1 - obj_ibw_in) / (ibw_1 - ibw_in)
                ibw_2 = (ibw_in * slope_ibw - .5 * grad_ibw_in * (ibw_1 + ibw_in)) / (slope_ibw - grad_ibw_in)
                obj_ibw_2 = _get_obj_ibandw(ibw_2)
                ibw_out = tf.cond( tf.reduce_all(tf.logical_and(tf.is_finite(ibw_2), tf.greater(ibw_2, 0.))),\
                    lambda: tf.cond(tf.less(obj_ibw_1, obj_ibw_in),\
                        lambda: tf.cond(tf.less(obj_ibw_2, obj_ibw_1), lambda: ibw_2, lambda: ibw_1),\
                        lambda: tf.cond(tf.less(obj_ibw_2, obj_ibw_in), lambda: ibw_2, lambda: ibw_in)),\
                    lambda: tf.cond(tf.less(obj_ibw_1, obj_ibw_in), lambda: ibw_1, lambda: ibw_in) )
                # ibw_out = tf.Print(ibw_out, [ibw_in, obj_ibw_in, grad_ibw_in, ibw_1, obj_ibw_1, ibw_2, obj_ibw_2, ibw_out])
                ##
                ibandw = ibandw.assign(ibw_out)
            else: raise ValueError('unknown "bwType": "{}"!'.format(self.pm.bwType))

        # optimization options
        if self.pm.optType == 'gd':
            stepsize = tf.constant(self.pm.stepsize, dtype=fltype)
        elif self.pm.optType == 'sgd':
            optIter0 = self.pm.optIter0 if hasattr(self.pm, 'optIter0') else 0
            stepsize = tf.constant(self.pm.stepsize, dtype=fltype) / ((global_step_fl + optIter0) ** self.pm.optExpo)
        elif self.pm.optType == 'adag':
            stepsize = tf.constant(self.pm.stepsize, dtype=fltype)
            with tf.variable_scope(var_scope, reuse=reuse):
                L_hisgrad = [tf.get_variable('hisgrad_' + get_name(samples), shape=samples.initialized_value().get_shape(), dtype=fltype, initializer=tf.zeros_initializer()) for samples in L_samples]
            # vecfld adjustment defined after vecfld is defined
        else: raise ValueError('unknown "optType": "{}"!'.format(self.pm.optType))

        # vecfld
        exp_ij = tf.exp(-.5*ibandw*dist2_ij)
        exp_i = tf.reduce_sum(exp_ij, axis=1)
        if self.dnType == 'GFSD':
            L_vecfld = [grad_logp + ibandw * \
                    ( particles - tf.tensordot(exp_ij/exp_i, particles, [[0], [0]]) )\
                    for grad_logp, particles in zip(L_grad_logp, L_particles)]
        elif self.dnType == 'GFSF':
#            L_shp = [tf.concat([[M], tf.ones([rk-1], intype)], 0) for rk in L_rk] # works even when rk is 1!
#            L_perm = [tf.range(rk) + tf.one_hot(0, rk, dtype=intype, on_value=rk-2) - tf.one_hot(rk-2, rk, dtype=intype, on_value=rk-2) for rk in L_rk]
#            chol_ij = tf.cholesky( exp_ij + self.pm.dnRidge*tf.reduce_max(exp_ij)*tf.eye(M.value, dtype=fltype) )
#            L_vecfld = [grad_logp + ibandw * \
#                    ( - particles + tf.transpose(\
#                        tf.cholesky_solve(\
#                            chol_ij, tf.transpose(tf.reshape(exp_i, shp)*particles, perm=perm)\
#                        ), perm=perm)\
#                    ) for grad_logp, particles, shp, perm in zip(L_grad_logp, L_particles, L_shp, L_perm)]
#
            L_shp = [tf.concat([[M], tf.ones([rk-1], intype)], 0) for rk in L_rk] # works even when rk is 1!
            L_perm = [tf.range(rk) + tf.one_hot(0, rk, dtype=intype, on_value=rk-2) - tf.one_hot(rk-2, rk, dtype=intype, on_value=rk-2) for rk in L_rk]
            exp_inv_ij = tf.linalg.inv( exp_ij + self.pm.dnRidge*tf.reduce_max(exp_ij)*tf.eye(M.value, dtype=fltype) )
            L_vecfld = [grad_logp + ibandw * \
                    ( - particles + tf.tensordot(exp_inv_ij, tf.reshape(exp_i, shp)*particles, axes=[[1],[0]])\
                    ) for grad_logp, particles, shp, perm in zip(L_grad_logp, L_particles, L_shp, L_perm)]
        elif self.dnType == 'SVGD':
            L_shp = [tf.concat([[M], tf.ones([rk-1], intype)], 0) for rk in L_rk]
            if self.pm.dnNormalize: coeff = tf.pow(ibandw/(2.*np.pi), .5*D_fl) / M_fl
            else: coeff = 1./M_fl
            L_vecfld = [coeff * ( tf.tensordot(exp_ij, grad_logp, [[1],[0]]) + ibandw * \
                    (tf.reshape(exp_i, shp)*particles - tf.tensordot(exp_ij, particles, [[1],[0]])) )\
                    for grad_logp, particles, shp in zip(L_grad_logp, L_particles, L_shp)]
        elif self.dnType == 'Blob':
            L_shp = [tf.concat([[M], tf.ones([rk-1], intype)], 0) for rk in L_rk]
            nor_exp_ij = exp_ij / exp_i
            L_vecfld = [grad_logp + ibandw * \
                    (particles - tf.tensordot((nor_exp_ij + tf.transpose(nor_exp_ij)), particles, [[0], [0]]) \
                    + tf.reshape(tf.reduce_sum(nor_exp_ij, axis=1), shp) * particles )\
                    for grad_logp, particles, shp in zip(L_grad_logp, L_particles, L_shp)]
        # MCMCs. Maybe a little troublesome to fit in this framework
        elif self.dnType == 'LD':
            if self.pm.optType not in {'gd', 'sgd'}: raise ValueError('For {}, "optType" should be either "gd" or "sgd" only!'.format(self.dnType));
            L_vecfld = [grad_logp + tf.sqrt(2./stepsize) * tf.random_normal(particles.initialized_value().get_shape(), dtype=fltype)\
                    for grad_logp, particles in zip(L_grad_logp, L_particles)]
        elif self.dnType == 'SGNHT':
            if self.pm.optType not in {'gd', 'sgd'}: raise ValueError('For {}, "optType" should be either "gd" or "sgd" only!'.format(self.dnType));
            with tf.variable_scope(var_scope, reuse=reuse):
                L_momentum = [tf.get_variable('momentum_' + get_name(particles), shape=particles.initialized_value().get_shape(), dtype=fltype, initializer=tf.random_normal_initializer()) for particles in L_particles]
                thermo = tf.get_variable('thermostats', initializer=tf.constant(self.pm.dnDiffusion, shape=[M], dtype=fltype))
            thermo = thermo.assign_add(\
                    stepsize*( tf.add_n([tf.reduce_sum(momentum**2, axis=rg) for momentum, rg in zip(L_momentum, L_rg)])/D_fl - 1. ) )
            L_shp = [tf.concat([[M], tf.ones([rk-1], intype)], 0) for rk in L_rk]
            L_vecfld = [momentum.assign_add( - stepsize*tf.reshape(thermo, shp)*momentum + stepsize*grad_logp + np.sqrt(2.*self.pm.dnDiffusion)*tf.sqrt(stepsize)*tf.random_normal(momentum.initialized_value().get_shape(), dtype=fltype) )
                    for momentum, grad_logp, shp in zip(L_momentum, L_grad_logp, L_shp)]
        elif self.dnType == 'SGRLD-lda-expand-natural':
            if self.pm.optType not in {'gd', 'sgd'}: raise ValueError('For {}, "optType" should be either "gd" or "sgd" only!'.format(self.dnType));
            L_vecfld = [tf.exp(-particles) * (grad_logp - 1.) + tf.sqrt(2./stepsize) * tf.exp(-.5*particles) * tf.random_normal(particles.initialized_value().get_shape(), dtype=fltype)\
                    for grad_logp, particles in zip(L_grad_logp, L_particles)]
        else: raise ValueError('unknown "dnType": "{}"!'.format(self.dnType))

        if self.pm.optType == 'adag':
            L_op_hisgrad = [hisgrad.assign(self.pm.optRemem * hisgrad + (1. - tf.cast(tf.sign(global_step-1), fltype) * self.pm.optRemem) * (vecfld ** 2))
                    for hisgrad, vecfld in zip(L_hisgrad, L_vecfld)]
            L_vecfld = [vecfld / (self.pm.optFudge + tf.sqrt(op_hisgrad)) for vecfld, op_hisgrad in zip(L_vecfld, L_op_hisgrad)]

        # acceleration options
        def inv_exp_w2(x, y):
            # _, _, vec = get_sinkhorn_lse(x, y, M, M, self.pm.accInvReg, self.pm.accMaxIter)
            _, _, vec = get_ipot_lse(x, y, M, M, self.pm.accInvReg, self.pm.accMaxIter)
            return vec

        L_op_particles = [None] * len(L_samples)
        with tf.control_dependencies(L_vecfld):

            if self.pm.accType == 'wgd':
                L_op_particles = [samples.assign_add(stepsize * vecfld) for samples, vecfld in zip(L_samples, L_vecfld)]

            elif self.pm.accType == 'po':
                L_op_diffsamples = [diffsamples.assign(\
                        self.pm.accRemem * diffsamples\
                        + stepsize * (\
                            vecfld + self.pm.accNoise / (global_step_fl ** self.pm.accExpo) * tf.random_normal(samples.initialized_value().get_shape(), dtype=fltype)\
                        )\
                    ) for diffsamples, samples, vecfld in zip(L_diffsamples, L_samples, L_vecfld)]
                L_op_particles = [samples.assign_add(op_diffsamples) for samples, op_diffsamples in zip(L_samples, L_op_diffsamples)]

            elif self.pm.accType == 'wnag':
                for i, (samples, auxsamples, oldsamples, vecfld) in enumerate(zip(L_samples, L_auxsamples, L_oldsamples, L_vecfld)):
                    op_oldsamples = oldsamples.assign(samples)
                    with tf.control_dependencies([op_oldsamples]):
                        op_samples = samples.assign(auxsamples + stepsize * vecfld)
                        L_op_particles[i] = auxsamples.assign(\
                                op_samples + (1.-1./global_step_fl) * (auxsamples - op_oldsamples)\
                                + (1. + (self.pm.accRemem-2.)/global_step_fl) * stepsize * vecfld )

            elif self.pm.accType == 'wnag-sink':
                for i, (samples, auxsamples, oldsamples, vecfld) in enumerate(zip(L_samples, L_auxsamples, L_oldsamples, L_vecfld)):
                    op_oldsamples = oldsamples.assign(samples)
                    with tf.control_dependencies([op_oldsamples]):
                        op_samples = samples.assign(auxsamples + stepsize * vecfld)
                        L_op_particles[i] = auxsamples.assign(\
                                op_samples - (1.-1./global_step_fl) * inv_exp_w2(auxsamples, op_oldsamples)\
                                + (1. + (self.pm.accRemem-2.)/global_step_fl) * stepsize * vecfld )

            elif self.pm.accType == 'wnes0':
                for i, (samples, auxsamples, oldsamples, vecfld) in enumerate(zip(L_samples, L_auxsamples, L_oldsamples, L_vecfld)):
                    op_oldsamples = oldsamples.assign(samples)
                    with tf.control_dependencies([op_oldsamples]):
                        op_samples = samples.assign(auxsamples + stepsize * vecfld)
                        L_op_particles[i] = auxsamples.assign(\
                                op_samples + (1.-3./(global_step_fl+2.)) * (op_samples - op_oldsamples))

            elif self.pm.accType == 'wnes1':
                muH = self.pm.accHessBnd * stepsize
                beta = self.pm.accShrink * tf.sqrt(muH)
                for i, (samples, auxsamples, oldsamples, vecfld) in enumerate(zip(L_samples, L_auxsamples, L_oldsamples, L_vecfld)):
                    op_oldsamples = oldsamples.assign(samples)
                    with tf.control_dependencies([op_oldsamples]):
                        op_samples = samples.assign(auxsamples + stepsize * vecfld)
                        L_op_particles[i] = auxsamples.assign(\
                                op_samples\
                                + (1 + beta - (2*(1+beta)*(2+beta)*muH) / (tf.sqrt(beta*beta + 4*(1+beta)*muH) - beta + 2*(1+beta)*muH)) * (op_samples - op_oldsamples))

            elif self.pm.accType == 'wnes1-sink':
                muH = self.pm.accHessBnd * stepsize
                beta = self.pm.accShrink * tf.sqrt(muH)
                alpha = .5 * (tf.sqrt(beta*beta + 4*(1+beta)*muH) - beta)
                gamma = self.pm.accHessBnd * (1. - 2.*beta / (tf.sqrt(beta*beta + 4*(1+beta)*muH) + beta))
                for i, (samples, auxsamples, oldsamples, vecfld) in enumerate(zip(L_samples, L_auxsamples, L_oldsamples, L_vecfld)):
                    op_oldsamples = oldsamples.assign(samples)
                    with tf.control_dependencies([op_oldsamples]):
                        op_samples = samples.assign(auxsamples + stepsize * vecfld)
                        L_op_particles[i] = auxsamples.assign(\
                                op_samples\
                                + alpha*gamma/(gamma+alpha*self.pm.accHessBnd) * inv_exp_w2(\
                                    op_samples,\
                                    auxsamples + (alpha-1.)/alpha * inv_exp_w2(auxsamples, op_oldsamples) + 1./alpha * inv_exp_w2(auxsamples, op_samples)\
                                ))

        return [tf.group(global_step, *L_op_particles), DynamicsInfo(L_samples, L_particles, L_grad_logp, L_vecfld, global_step, 1./ibandw, ibandw, bwmed, bwmax, var_scope)]

