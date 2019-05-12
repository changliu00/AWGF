class HP(object):
    n_repeat = 20; n_round = 8; n_iter = 1000
    n_hidden = 50; M = 20
    batchsize = 100

class PM(object):
    dtFile = '4kin8nm.txt'

    dnType='SVGD'; dnNormalize=False; accType='wgd'; optType='adag'; optRemem=.9; optFudge=1e-6; stepsize=1e-3; bwType='med'
    # dnType='SVGD'; dnNormalize=False; accType='po'; accExpo=1.; accRemem=.6; accNoise=1e-7; optType='adag'; optRemem=.9; optFudge=1e-6; stepsize=1e-4; bwType='med'
    # dnType='SVGD'; dnNormalize=False; accType='wnag'; accRemem=3.6; optType='adag'; optRemem=.9; optFudge=1e-6; stepsize=1e-6; bwType='med'
    # dnType='SVGD'; dnNormalize=False; accType='wnes1'; accHessBnd=1000.; accShrink=.2; optType='adag'; optRemem=.9; optFudge=1e-6; stepsize=1e-4; bwType='med'

    # dnType='Blob'; accType='wgd'; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='Blob'; accType='po'; accExpo=1.; accRemem=.8; accNoise=1e-7; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='Blob'; accType='wnag'; accRemem=3.5; optType='sgd'; optExpo=.5; stepsize=1e-5; bwType='med'
    # dnType='Blob'; accType='wnes1'; accHessBnd=3000.; accShrink=.2; optType='sgd'; optExpo=.6; stepsize=1e-4; bwType='med'

    # dnType='GFSD'; accType='wgd'; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='GFSD'; accType='po'; accExpo=1.; accRemem=.8; accNoise=1e-7; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='GFSD'; accType='wnag'; accRemem=3.5; optType='sgd'; optExpo=.5; stepsize=1e-5; bwType='med'
    # dnType='GFSD'; accType='wnes1'; accHessBnd=3000.; accShrink=.2; optType='sgd'; optExpo=.6; stepsize=1e-4; bwType='med'

    # dnType='GFSF'; dnRidge=1e-2; accType='wgd'; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='GFSF'; dnRidge=1e-2; accType='po'; accExpo=1.; accRemem=.8; accNoise=1e-7; optType='sgd'; optExpo=.5; stepsize=3e-5; bwType='med'
    # dnType='GFSF'; dnRidge=1e-2; accType='wnag'; accRemem=3.5; optType='sgd'; optExpo=.5; stepsize=1e-5; bwType='med'
    # dnType='GFSF'; dnRidge=1e-2; accType='wnes1'; accHessBnd=3000.; accShrink=.2; optType='sgd'; optExpo=.6; stepsize=1e-4; bwType='med'


