import socket, sys, time

import numpy
from numpy import asarray, random
random.seed(2344)

from theano.tensor import lscalar, lvector, matrix, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano import shared, function, config
import theano

def rand(*size):
    return asarray(random.rand(*size), dtype=config.floatX)
def randn(*size):
    return asarray(random.randn(*size), dtype=config.floatX)
def randint(size, high):
    return asarray(random.randint(size=size, low=0, high=high), dtype='int32')
def zeros(*size):
    return numpy.zeros(size, dtype=config.floatX)

verbose=0
n_examples=1000
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

data_x = shared(randn(n_examples, 1, 32, 32))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si+nsi]
sy = data_y[si:si+nsi]

bmark = None

if config.floatX == 'float32':
    prec = 'float'
else:
    prec = 'double'

def reportmodel(model, batchsize, v, extra=''):
    bmark.write("%s\t" % model)
    bmark.write("theano{%s/%s/%i%s}\t" % (
        config.device[0:3], prec, batchsize, extra))
    bmark.write("%.2f\n"%v)

def eval_and_report(train, name, batchsizes, N=n_examples, extra=""):
    for bs in batchsizes:
        assert N % bs == 0 # can't be cheatin now...
        t = time.time()
        for i in xrange(N/bs):
            cost = train(i*bs, bs)
            if not (i % (1000/bs)):
                print i*bs, cost
        reportmodel(name, bs, N/(time.time()-t), extra=extra)

def bench_ConvSmall(batchsize):
    data_x.value = randn(n_examples, 1, 32, 32)
    w0 = shared(rand(6, 1, 5, 5) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 5, 5) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*5*5, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 32, 32), filter_shape=(6, 1, 5, 5), verbose=verbose) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool_2d(c0, (2,2))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 14, 14), filter_shape=(16,6,5,5), verbose=verbose) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (2,2)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])

    eval_and_report(train, "ConvSmall", [batchsize], N=600)

def bench_ConvMed(batchsize, extra=""):
    data_x.value = randn(n_examples, 1, 96, 96)
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*8*8, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 96, 96), filter_shape=(6,1,7,7), verbose=verbose) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool_2d(c0, (3,3))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 30, 30), filter_shape=(16,6,7,7), verbose=verbose) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (3,3)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    eval_and_report(train, "ConvMed", [batchsize], N=120, extra=extra)

def bench_ConvLarge(batchsize, extra=""):
    data_x.value = randn(n_examples, 1, 256, 256)
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*11*11, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 256, 256), filter_shape=(6,1,7,7), verbose=verbose) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool_2d(c0, (5,5))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 50, 50), filter_shape=(16,6,7,7), verbose=verbose) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (4,4)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    eval_and_report(train, "ConvLarge", [batchsize], N=120, extra=extra)

if __name__ == '__main__':
    fft=False
    fft_valid = False
    no_orig =False
    for param in sys.argv[1:]:
        if param == '--fft':
            fft=True
        elif param == '--fft-valid':
            fft_valid = True
        elif param == '--append':
            bmark = open("%s_convnet_%s_%s.bmark"% (socket.gethostname(), config.device, config.floatX), 'a')
        elif param.startswith('--verbose='):
            verbose = int(param[10:])
        elif param == '--no_orig':
            no_orig=True
        else:
            print "Unknow parameter",param
            sys.exit(1)

    if bmark is None:#
        bmark = open("%s_convnet_%s_%s.bmark"% (socket.gethostname(), config.device, config.floatX), 'w')

    types = []
    if no_orig == False:
        types.append('orig')
    if fft:
        types.append('fft')
    if fft_valid:
        types.append('fft_valid')
    for type in types:
        extra = ''
        if type == 'fft':
            print "\n\n\n WILL BE USING GpuFFTConvOp for full gpu convolution\n\n\n"
            from fft_conv_op import fft_conv_op
            theano.config.GpuFFTConvOp.valid = False
            extra = '/fft'
        if type == 'fft_valid':
            print "\n\n\n WILL BE USING GpuFFTConvOp for full and valid gpu convolution\n\n\n"
            from fft_conv_op import fft_conv_op
            theano.config.GpuFFTConvOp.valid = True
            extra = "/fft-valid"

        bench_ConvSmall(1)
        bench_ConvSmall(60)
        bench_ConvMed(1, extra=extra)
        bench_ConvMed(60)
        bench_ConvLarge(1, extra=extra)
        bench_ConvLarge(60)


