import time, socket
import numpy
import scipy.signal
from numpy import asarray, random
random.seed(2344)

def rand(*size):
    return asarray(random.rand(*size),dtype = 'float64')
def randn(*size):
    return asarray(random.randn(*size), dtype= 'float64')
def randint(size, high):
    return asarray(random.randint(size=size, low=0, high=high), dtype='int32')
def zeros(*size):
    return numpy.zeros(size, dtype='float64')

n_examples=1000
outputs=10

data_x = randn(n_examples, 1, 32, 32)
data_y = randint((n_examples,), outputs)

#si = lscalar()
#nsi = lscalar()
#sx = data_x[si:si+nsi]
#sy = data_y[si:si+nsi]

bmark = open("%s_convnet_%s_%s.bmark"% (socket.gethostname(), 'cpu', 'float64'), 'w')

if 'float64' == 'float32':
    prec = 'float'
else:
    prec = 'double'

def reportmodel(model, batchsize, v):
    bmark.write("%s\t" % model)
    bmark.write("scipy{%s/%s/%i}\t" % (
        'cpu', prec, batchsize))
    bmark.write("%.2f\n"%v)

def eval_and_report(train, name, batchsizes, N=n_examples):
    for bs in batchsizes:
        assert N % bs == 0 # can't be cheatin now...
        t = time.time()
        for i in xrange(N/bs):
            cost = train(i*bs, bs)
            if not (i % (1000/bs)):
                print i*bs, cost
        reportmodel(name, bs, N/(time.time()-t))

def bench_ConvSmall(batchsize):
    data_x = randn(n_examples, 32, 32)
    w0 = rand(6, 5, 5) * numpy.sqrt(6 / (25.))
    w1 = rand(16*6, 5, 5) * numpy.sqrt(6 / (25.))

    def train(si, nsi):
        dx = data_x[si]
        dy = data_x[si,:16,:16]
        for i in xrange(6):
            c0 = numpy.tanh(scipy.signal.convolve2d(dx,w0[i]))
        for j in xrange(6*16):
            c1 = numpy.tanh(scipy.signal.convolve2d(dy,w1[j]))

    eval_and_report(train, "ConvSmall", [batchsize], N=600)

def bench_ConvMed(batchsize):
    data_x = randn(n_examples, 96, 96)
    w0 = rand(6, 7, 7) * numpy.sqrt(6 / (25.))
    w1 = rand(16*6, 7, 7) * numpy.sqrt(6 / (25.))

    def train(si, nsi):
        dx = data_x[si]
        dy = data_x[si,:48,:48]
        for i in xrange(6):
            c0 = numpy.tanh(scipy.signal.convolve2d(dx,w0[i],mode='same'))
        for j in xrange(6*16):
            c1 = numpy.tanh(scipy.signal.convolve2d(dy,w1[j], mode='same'))

    eval_and_report(train, "ConvMed", [batchsize], N=120)

def bench_ConvLarge(batchsize):
    data_x = randn(n_examples, 256, 256)
    w0 = rand(6* 1, 7, 7) * numpy.sqrt(6 / (25.))
    w1 = rand(16* 6, 7, 7) * numpy.sqrt(6 / (25.))

    def train(si, nsi):
        dx = data_x[si]
        dy = data_x[si,:48,:48]
        for i in xrange(6):
            c0 = numpy.tanh(scipy.signal.convolve2d(dx,w0[i],mode='same'))
        for j in xrange(6*16):
            c1 = numpy.tanh(scipy.signal.convolve2d(dy,w1[j],mode = 'same'))

    eval_and_report(train, "ConvLarge", [batchsize], N=120)

if __name__ == '__main__':
    bench_ConvSmall(1)
    #bench_ConvSmall(60)
    bench_ConvMed(1)
    #bench_ConvMed(60)
    bench_ConvLarge(1)
    #bench_ConvLarge(60)

