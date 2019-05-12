class HP(object):
    dtName = 'icml'; dtFilename = './data/icml.txt'; dtVocname = './data/icml.voc'
    alpha=1e-1; beta=1e-1; sigma=1.
    K=30; batchsize=100; n_gsamp=50; M=20
    n_iter=20; n_round=25
    # n_iter=100; n_round=100
    perpType = 'para'; n_window = None

class PM(object):
    accType = 'wgd';
    # dnType='SVGD'; dnNormalize=False; optType='sgd'; optExpo=.55; optIter0=1000; stepsize=3e+0; bwType='med'
    # dnType='SVGD'; dnNormalize=False; optType='sgd'; optExpo=.55; optIter0=1; stepsize=1e-1; bwType='he'; bwSubType='h'

    # dnType='Blob'; optType='sgd'; optExpo=.55; optIter0=1000; stepsize=3e-1; bwType='med'

    # dnType='GFSD'; optType='sgd'; optExpo=.55; optIter0=1000; stepsize=3e-1; bwType='med'

    # dnType='GFSF'; dnRidge=1e-5; optType='sgd'; optExpo=.55; optIter0=1000; stepsize=3e-1; bwType='med'

    # dnType='SGNHT'; dnDiffusion=22.4; optType='gd'; stepsize=4.47e-3
    dnType='SGNHT'; dnDiffusion=22.4; optType='gd'; stepsize=3e-2

