"""Code to plot graphs of benchmarking for scipy paper
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

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


    dirs={}

    def parse_bmark(args, dirname, fnames):
        d={}
        for f in fnames:
            if f.endswith('.bmark'):
                lines = open(os.path.join(dirname,f)).readlines()
                for line in lines:
                    line_ = line.strip()
                    if not line_: continue
                    sp = line_.split('\t')
                    d[tuple(sp[0:2])] = float(sp[2])
        dirs[dirname]=d
    
    os.path.walk('numpy',parse_bmark, None)
    #not used
    #('mlp_32_10', 'numpy{1}'): ['14229.80\n'],
    #('mlp_32_10', 'numpy{60}'): ['369227.74\n'],

    os.path.walk('scipy',parse_bmark, None)
    #not used
    #('ConvSmall', 'scipy{cpu/double/1}'): ['47.08'],
    #('ConvMed', 'scipy{cpu/double/1}'): ['5.20'],

    os.path.walk('theano',parse_bmark, None)
    os.path.walk('matlab',parse_bmark, None)
    os.path.walk('torch5',parse_bmark, None)
    os.path.walk('eblearn',parse_bmark, None)

    import pdb;pdb.set_trace()

    def get(dest, key, source, key2):
        try:
            dest[key] = source[key2]
        except KeyError:
            print "missing key", key2

    def gets(backend, key, key2):
        get(conv, backend+key, dirs[backend], ('ConvLarge',backend+key2))
        get(logreg, backend+key, dirs[backend], ('mlp_784_10',backend+key2))
        get(s_mlp, backend+key, dirs[backend], ('mlp_784_500_10',backend+key2))
        get(d_mlp, backend+key, dirs[backend], ('mlp_784_1000_1000_1000_10',backend+key2))
        get(rbm, backend+key, dirs[backend], ('cd1 rbm_bernoulli 1024_1024',backend+key2))
    #comments are conv#logreg#s_mlp#d_mlp#rbm
    gets('matlab','[1][cpu]','{cpu/double/1}')#?#15510.25#316.72#21.87#12.81
    gets('matlab','[60][cpu]','{cpu/double/60}')#?#40778.05#3285.53#404.90#288.77
    gets('matlab','[1][gpu]','{gpu/float/1}')#?#214.67#122.69#51.03#?
    gets('matlab','[60][gpu]','{gpu/float/60}')#?#10378.20#5809.93#1876.25#?
    gets('numpy','[1]','{1}')#?#3060.13#110.27#?#21.34
    gets('numpy','[60]','{60}')#?#38370.49#2523.21#?#352.38
    gets('theano','[1][cpu]','{cpu/double/1}')#10.92#4299.02#1036.43#32.10#79.24
    gets('theano','[60][cpu]','{cpu/double/60}')#10.23#79729.01#4902.42#483.75#356.91
    gets('theano','[1][gpu]','{gpu/float/1}')#79#1850.63#793.69#229.85#498.29
    gets('theano','[60][gpu]','{gpu/float/60}')#78.44#70855.12#38310#6882.59#12598.77
    gets('torch5','[1][cpu]','')#5.70#logreg#s_mlp#d_mlp#rbm
    gets('torch5','[60][cpu]','')#5.70#logreg#770.22#47.61#rbm

    conv['eblearn']                  = dirs['eblearn'][('ConvLarge', 'eblearn')]#6.31361
    conv['eblearn_ipp']              = dirs['eblearn'][('ConvLarge', 'eblearn{ipp}')]#6.15147
    conv['scipy[1]']                 = dirs['scipy'][('ConvLarge', 'scipy{cpu/double/1}')]#2.37

    #logreg['pybrain']                = 1096.11
    #logreg['pybrain[arac]']          = 1843.11
    #s_mlp['pybrain']                 = 45.22
    #s_mlp['pybrain[arac]']           = 61.09
    #s_mlp['matlab_nn[1]']            = 29.52
    #s_mlp['matlab_nn[60]']           = 1058.43
    #d_mlp['pybrain']                 = 6.02
    #d_mlp['pybrain[arac]']           = 7.07
    #d_mlp['matlab_nn[1]']            = 4.19
    #d_mlp['matlab_nn[60]']           = 172.11
    #rbm['cudamat[1]']                = 460.30
    #rbm['cudamat[60]']               = 13214.87


    times['logreg']      = logreg
    times['shallow_mlp'] = s_mlp
    times['deep_mlp']    = d_mlp
    times['conv_net']    = conv
    times['rbm']         = rbm


    # Plot 1 : Shallow MLP  ( only batch size 60 )
    names = [\
             ('theano[60][gpu]' ,'Theano using the GPU, 38310 examples/sec','GPU', 'r') \
             , ('matlab[60][gpu]' ,'Matlab using the GPU, 5809 examples/sec','GPU','k') 
             , ('theano[60][cpu]' ,'Theano using the CPU, 4902 examples/sec','CPU','g') \
             , ('torch5[60][cpu]'  ,'Torch, 770 examples/sec','CPU','b')                \
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
    names = [
        ('theano[1][gpu]' , 'Theano using the GPU, 79 examples/sec', 'GPU', 'r'),
        ('theano[1][cpu]' , 'Theano using the CPU, 10 examples/sec','CPU','g'),
        ('torch5[1][cpu]'  , 'Torch, 5 examples/sec', 'CPU', 'b'),
        ('scipy[1]'       , 'SciPy, 2 examples/sec *','CPU','y'),
        ('eblearn'        , 'EBLearn, 6 examples/sec','CPU','c'),
        ]

    name_vals = [ x[1] for x in names]
    vals      = [ times['conv_net'][x[0]] for x in names]
    tps       = [ x[2] for x in names ] 
    cols      = [ x[3] for x in names ]
    figure(vals, name_vals, tps, cols, save='conv.pdf')
