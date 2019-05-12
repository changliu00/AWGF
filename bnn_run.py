import tensorflow as tf
import numpy as np
from dynamics import Dynamics
from bnn_def import BayesNN, Dataset
from functools import partial
import time, os, sys

def merge_dicts(*dicts):
    return {k:v for d in dicts for k, v in d.items()}

def vars_stat(obj):
    return {k:getattr(obj, k) for k in dir(obj) if not k.startswith('_')}

if __name__ == '__main__':
    exec('from ' + sys.argv[1][:-3] + ' import HP, PM')
    hp = HP(); pm = PM()
    print('settings file loaded')

    np.random.seed(1)
    dataFile = './data/' + pm.dtFile
    data = Dataset(dataFile, batchsize=hp.batchsize, train_ratio=.9, dev_ratio=.1)
    print('Dataset "{}" loaded.'.format(dataFile))
    print('featsize: {:d}, trainsize: {:d}, testsize: {:d}'.format(data.featsize, data.trainsize, data.testsize))
    model = BayesNN(featsize = data.featsize, M = hp.M, n_hidden = hp.n_hidden, Y_std=data.Y_std) ##
    op_samples, dninfo = Dynamics(pm.dnType, pm).evolve(model.latvar, get_logp = partial(model.get_logp, fullsize=data.trainsize))
    
    T_rmse = np.zeros([hp.n_repeat, hp.n_round])
    T_llh = np.zeros([hp.n_repeat, hp.n_round])
    L_time = np.zeros([hp.n_repeat])
    for i in range(hp.n_repeat):
        if i != 0: data.reset()
        print('Repeat-trial {:d}:'.format(i))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            X_n, Y_n = data.get_batch_for_init_loggamma()
            sess.run(model.init_loggamma, {model.X_train: X_n*data.X_std+data.X_mean, model.Y_train: Y_n*data.Y_std+data.Y_mean})
            for j in range(hp.n_round):
                t0 = time.time()
                for k in range(hp.n_iter):
                    sess.run(op_samples, dict(zip((model.X_train, model.Y_train), data.get_batch() )))
                L_time[i] += time.time() - t0
                T_rmse[i,j], T_llh[i,j] = sess.run([model.rmse, model.llh], {
                    model.X_dev: data.X_dev, model.Y_dev: data.Y_dev,
                    model.X_test: data.X_test, model.Y_test: data.Y_test})
                T_rmse[i,j] *= data.Y_std
                print('iteration {:5d}: rmse {:.3e}, llh {:.3e}, time {:.2f}'.format((j+1)*hp.n_iter, T_rmse[i,j], T_llh[i,j], L_time[i]))

    L_rmse = T_rmse[:,-1]; L_llh = T_llh[:,-1]
    rmse_mean = np.mean(L_rmse); rmse_std = np.std(L_rmse)
    llh_mean = np.mean(L_llh); llh_std = np.std(L_llh)
    time_mean = np.mean(L_time)
    print('Summary: rmse {:.3e} pm {:.3e}, llh {:.3e} pm {:.3e}, time {:.2f}'.format(rmse_mean, rmse_std, llh_mean, llh_std, time_mean))
    
    resDir = './bnn_res_' + pm.dtFile[:-4] + '/'
    if not os.path.isdir(resDir): os.makedirs(resDir)
    resFile_root = resDir + '_'.join([pm.dnType, pm.accType, pm.bwType if hasattr(pm, 'bwType') else 'void', pm.optType])
    appd = -1
    while True:
        appd += 1; resFile = resFile_root + '_{:d}.npz'.format(appd)
        if not os.path.exists(resFile): break
    print('Writing results to file "{}"'.format(resFile))
    np.savez(resFile,
            T_rmse = T_rmse, T_llh = T_llh, L_time = L_time,
            rmse_mean = rmse_mean, rmse_std = rmse_std, llh_mean = llh_mean, llh_std = llh_std, time_mean = time_mean,
            **merge_dicts(vars_stat(hp), vars_stat(pm)))

