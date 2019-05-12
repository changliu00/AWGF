from __future__ import division

import numpy as np
import tensorflow as tf
import os, sys, time
from lda_def import LDA, Dataset
from dynamics import Dynamics
# np.set_printoptions(precision=3, suppress=True)

def merge_dicts(*dicts):
    return {k:v for d in dicts for k, v in d.items()}

def vars_stat(obj):
    return {k:getattr(obj, k) for k in dir(obj) if not k.startswith('_')}

if __name__ == '__main__':
    exec('from ' + sys.argv[1][:-3] + ' import HP, PM')
    args = HP(); pm = PM()
    print(vars_stat(pm))

    data = Dataset(args.dtFilename, args.batchsize)
    data_W = sum(1 for line in open(args.dtVocname) if line.rstrip())
    model = LDA(data.n_tr, data_W, args.K, args.alpha, args.beta, args.sigma, args.n_gsamp)
    model.set_holdout_logperp(args.perpType, data.ho_train_cts, data.ho_test_cts, args.n_window)
    theta = args.beta + args.sigma * np.random.normal(size=(args.M, args.K, data_W))
    theta_tf = tf.Variable(theta)
    grads_tf = tf.placeholder(dtype=theta.dtype, shape=theta.shape)
    op_samples, dninfo = Dynamics(pm.dnType, pm).evolve(theta_tf, L_grad_logp=grads_tf)
    tr_times = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        theta_smp, theta_par = zip( *sess.run([dninfo.L_samples, dninfo.L_particles]) )[0]
        for i in range(args.n_round):
            t_start = time.time()
            for j in range(args.n_iter):
                tr_train_cts, tr_test_cts = data.get_batch()
                grads = model.get_grad_logp(tr_train_cts, theta=theta_par)
                if j == args.n_iter-1: break
                theta_par = sess.run([op_samples, dninfo.L_particles], {grads_tf: grads})[1][0]
            theta_smp, theta_par = zip( *(sess.run([op_samples, dninfo.L_samples, dninfo.L_particles], {grads_tf: grads})[1:]) )[0]
            tr_times.append(time.time() - t_start + (tr_times[-1] if tr_times else 0.))
            # tr_logperp, phi_smp = model.get_training_logperp(tr_train_cts, tr_test_cts, theta=theta_smp)
            # ho_logperp, _ = model.get_holdout_logperp(phi=phi_smp)
            # print('iter: {:2d}, epoch: {:.3f}, tr_logperp: {:.3e}, ho_logperp: {:.3e}, time: {:.3f}'.format(model.nIter, model.epoch, tr_logperp, ho_logperp, tr_times[-1]))
            ho_logperp = model.get_holdout_logperp(theta=theta_smp)
            ho_logperp *= data.n_ho; model.ho_logperps[-1] *= data.n_ho
            print('iter: {:4d}, epoch: {:9.3f}, ho_logperp: {:.3e}, time: {:.3f}'.format(model.nIter, model.epoch, ho_logperp, tr_times[-1]))

    resDir = 'lda_res_' + args.dtName + '_' + args.perpType + '/'
    if not os.path.isdir(resDir): os.makedirs(resDir)
    resFile_root = resDir + '_'.join([pm.dnType, pm.accType, pm.bwType if hasattr(pm, 'bwType') else 'void', pm.optType])
    appd = -1
    while True:
        appd += 1; resFile = resFile_root + '_{:d}.npz'.format(appd)
        if not os.path.exists(resFile): break
    print('Writing results to file "{}"'.format(resFile))
    np.savez(resFile, tr_times=tr_times, **merge_dicts(vars_stat(args), vars_stat(pm), model.save_dict()))

