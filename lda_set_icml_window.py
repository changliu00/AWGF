class HP(object):
    dtName = 'icml'; dtFilename = './data/icml.txt'; dtVocname = './data/icml.voc'
    alpha=1e-1; beta=1e-1; sigma=1.
    K=30; batchsize=100; n_gsamp=50; M=1
    n_iter=20; n_round=25
    # n_iter=100; n_round=500
    perpType = 'window'; n_window = 20

class PM(object):
    # dnType='SGNHT'; dnDiffusion=22.4; accType='wgd'; optType='gd'; stepsize=4.47e-3
    dnType='SGNHT'; dnDiffusion=22.4; accType='wgd'; optType='gd'; stepsize=3e-2

