"""Code to plot graphs of benchmarking for scipy paper
"""

import numpy as np
import matplotlib.pyplot as plt


def figure(values, names, types, cols,
         values2 = None, names2 = None, save=None):
    """Bar plot of values and names    """
    plt.clf()
    #plt.subplot(121)
    barGroups(np.array(values), names, types, cols)
    plt.ylabel('Data entries per second')
    #plt.ylim([0, 18])

    #plt.subplot(122)
    #barGroups(np.array(values2), names2)
    #plt.ylabel('Data entries per second')
    #plt.ylim([500,3000])

    if save:
        plt.savefig(save, format="pdf")


def barGroups(A, names,types, colors):
    #colors = 'rgbycmkw'
    width = 1.0 / (A.shape[0] + 1)
    #plt.clf()
    ind = np.arange(A.shape[0])
    bars = []
    for p in range(A.shape[0]):
        bars += [plt.bar(p+0.1, A[p], color =colors[p])]
    plt.xticks(np.arange(A.shape[0])+0.5, types)
    #bar_groups = []
    #for c in range(A.shape[1]):
    #    bars = plt.bar(ind+c*width, A[:,c], width, yerr=S[:,c], 
    #                   color=colors[c % len(colors)])
    #    bar_groups.append(bars)
    #plt.xticks(ind+width, names)
    plt.legend(bars, names)


if __name__ == '__main__':

    # times 
    times  = {}
    logreg = {}
    s_mlp  = {}
    d_mlp  = {}
    conv   = {}
    rbm    = {}

    conv['eblearn']                  = 6.31361
    conv['eblearn_ipp']              = 6.15147
    conv['theano[1][cpu]']           = 10.92
    conv['theano[60][cpu]']          = 10.23
    #conv['theano[1][gpu]']           = 57.71
    conv['theano[1][gpu]']           = 79
    conv['theano[6][gpu]']           = 78.44
    conv['torch[1]']                 = 5.70
    conv['scipy[1]']                 = 2.37
    logreg['matlab[1][cpu]']         = 15510.25
    logreg['matlab[60][cpu]']        = 40778.05
    logreg['matlab[1][gpu]']         = 214.67
    logreg['matlab[60][gpu]']        = 10378.20
    logreg['numpy[1]']               = 3060.13
    logreg['numpy[60]']              = 38370.49
    logreg['pybrain']                = 1096.11
    logreg['pybrain[arac]']          = 1843.11
    logreg['theano[1][cpu]']         = 4299.02
    logreg['theano[60][cpu]']        = 79729.01
    logreg['theano[1][gpu]']         = 1850.63
    logreg['theano[60][gpu]']        = 70855.12


    s_mlp['theano[1][cpu]']          = 1036.43
    s_mlp['theano[60][cpu]']         = 4902.42
    s_mlp['theano[1][gpu]']          = 793.69
    #s_mlp['theano[60][gpu]']         = 30533.60
    s_mlp['theano[60][gpu]']         = 38310
    s_mlp['pybrain']                 = 45.22
    s_mlp['pybrain[arac]']           = 61.09
    s_mlp['matlab[1][cpu]']          = 316.72
    s_mlp['numpy[1]']                = 110.27
    s_mlp['numpy[60]']               = 2523.21
    s_mlp['matlab[60][cpu]']         = 3285.53
    s_mlp['matlab[1][gpu]']          = 122.69
    s_mlp['matlab[60][gpu]']         = 5809.93
    s_mlp['matlab_nn[1]']            = 29.52
    s_mlp['matlab_nn[60]']           = 1058.43
    s_mlp['torch[60]']               = 770.22

    d_mlp['torch[60]']               = 47.61
    d_mlp['theano[1][cpu]']          = 32.10
    d_mlp['theano[60][cpu]']         = 483.75
    d_mlp['theano[1][gpu]']          = 229.85
    d_mlp['theano[60][gpu]']         = 6882.59
    
    d_mlp['pybrain']                 = 6.02
    d_mlp['pybrain[arac]']           = 7.07
    
    d_mlp['matlab[1][cpu]']          = 21.87
    d_mlp['matlab[1][gpu]']          = 51.03
    d_mlp['matlab[60][cpu]']         = 404.90
    d_mlp['matlab[60][gpu]']         = 1876.25
    
    d_mlp['matlab_nn[1]']            = 4.19
    d_mlp['matlab_nn[60]']           = 172.11
    
    rbm['matlab[1][cpu]']            = 12.81
    rbm['matlab[60][cpu]']           = 288.77
    rbm['numpy[1]']                  = 21.34
    rbm['numpy[60]']                 = 352.38
    rbm['theano[1][cpu]']            = 79.24
    rbm['theano[60][cpu]']           = 356.91
    rbm['theano[1][gpu]']            = 498.29
    rbm['theano[60][gpu]']           = 12598.77
    rbm['cudamat[1]']                = 460.30
    rbm['cudamat[60]']               = 13214.87



    times['logreg']      = logreg
    times['shallow_mlp'] = s_mlp
    times['deep_mlp']    = d_mlp
    times['conv_net']    = conv
    times['rbm']         = rbm


    # Plot 1 : Deep MLP  ( only batch size 60 )
    names = [\
             ('theano[60][gpu]' ,'Theano using the GPU, 38310 examples/sec','GPU', 'r') \
             , ('matlab[60][gpu]' ,'Matlab using the GPU, 5809 examples/sec','GPU','k') 
             , ('theano[60][cpu]' ,'Theano using the CPU, 4902 examples/sec','CPU','g') \
             , ('torch[60]'       ,'Torch, 770 examples/sec','CPU','b')                \
             , ('numpy[60]'       ,'Numpy, 2523 examples/sec','CPU','y')
             , ('matlab[60][cpu]' ,'Matlab using the CPU, 3285 examples/sec','CPU','m')\
             #, ('pybrain'        ,'PyBrain')              \
             #, ('pybrain[arac]'  ,'PyBrain using Arac')   \
             ]
    name_vals = [ x[1] for x in names]
    vals      = [ times['shallow_mlp'][x[0]] for x in names ]
    tps       = [ x[2] for x in names ] 
    cols      = [ x[3] for x in names ]

    figure(vals, name_vals,  tps, cols,save = 'mlp.pdf')

    # Plot 2 : CONV
    names = [ \
             ('theano[1][gpu]' , 'Theano using the GPU, 79 examples/sec', 'GPU', 'r') \
             , ('theano[1][cpu]' , 'Theano using the CPU, 10 examples/sec','CPU','g') \
             , ('torch[1]'       , 'Torch, 5 examples/sec', 'CPU', 'b')
             , ('scipy[1]'       , 'SciPy, 2 examples/sec *','CPU','y')
             , ('eblearn'        , 'EBLearn, 6 examples/sec','CPU','c')
             ]

    name_vals = [ x[1] for x in names]
    vals      = [ times['conv_net'][x[0]] for x in names]
    tps       = [ x[2] for x in names ] 
    cols      = [ x[3] for x in names ]
    figure(vals, name_vals, tps, cols, save='conv.pdf')
