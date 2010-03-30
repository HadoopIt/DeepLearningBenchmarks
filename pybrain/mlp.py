import time
from numpy import asarray, random
from pybrain.utilities              import percentError
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.structure.modules      import SoftmaxLayer
from pybrain.datasets               import SupervisedDataSet

random.seed(2344)
n_examples = 10000


def rand(*size):
    return asarray(random.rand(*size))

def randn(*size):
    return asarray(random.randn(*size))

def zeros(*size):
    return numpy.zeros(size)

def large_data():
    data = SupervisedDataSet(784,10)
    for idx in xrange(n_examples):
        data.addSample( randn(784), randn(10)) 
    return data

def small_data():
    data = SupervisedDataSet(32,10)
    for idx in xrange(n_examples):
        data.addSample( randn(32), randn(10))
    return data


bmark = open('mlp_pybrain.bmark','w')

def benchmarking(arac = False):
    data32  = small_data()

    # mlp_32_10
    fnn = buildNetwork(32,10, outclass = SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset = data32, learningrate = 0.01, verbose = True)
    t = time.time()
    trainer.trainEpochs(1)
    t = time.time() - t
    bmark.write('mlp_32_10\t')
    bmark.write('pybrain_python atlas=OS_version\t')
    bmark.write('%.2f\n'%t)
    bmark.write('# Pybrain does not support NLL, squared error used \n')
    bmark.write('# Pybrain does not support minibatches \n')

    if arac: 
        del fnn
        del trainer
        fnn = buildNetwork(32,10, outclass = SoftmaxLayer, fast = True)
        trainer = BackpropTrainer(fnn, dataset = data32, learningrate = 0.01, verbose = True)
        t = time.time()
        trainer.trainEpochs(1)
        t = time.time() - t
        bmark.write('mlp_32_10\t')
        bmark.write('pybrain_arac atlas=OS_version\t')
        bmark.write('%.2f\n'%t)
        bmark.write('# Pybrain does not support NLL, squared error used \n')
        bmark.write('# Pybrain does not support minibatches \n')

    del data32
    del fnn
    del trainer

    data784 = large_data()
    fnn = buildNetwork(784,10, outclass = SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
    t = time.time()
    trainer.trainEpochs(1)
    t = time.time() - t
    bmark.write('mlp_784_10\t')
    bmark.write('pybrain_python atlas=OS_version\t')
    bmark.write('%.2f\n'%t)
    bmark.write('# Pybrain does not support NLL, squared error used \n')
    bmark.write('# Pybrain does not support minibatches \n')

    del fnn
    del trainer


    fnn = buildNetwork(784,500,10, outclass = SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
    t = time.time()
    trainer.trainEpochs(1)
    t = time.time() - t
    bmark.write('mlp_784_500_10\t')
    bmark.write('pybrain_python atlas=OS_version\t')
    bmark.write('%.2f\n'%t)
    bmark.write('# Pybrain does not support NLL, squared error used \n')
    bmark.write('# Pybrain does not support minibatches \n')


    del fnn
    del trainer


    fnn = buildNetwork(784,1000,1000,1000,10, outclass = SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
    t = time.time()
    trainer.trainEpochs(1)
    t = time.time() - t
    bmark.write('mlp_784_1000__1000_1000_10\t')
    bmark.write('pybrain_python atlas=OS_version\t')
    bmark.write('%.2f\n'%t)
    bmark.write('# Pybrain does not support NLL, squared error used \n')
    bmark.write('# Pybrain does not support minibatches \n')

    if arac:
        del fnn
        del trainer

        fnn = buildNetwork(784,10, outclass = SoftmaxLayer, fast = True)
        trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
        t = time.time()
        trainer.trainEpochs(1)
        t = time.time() - t
        bmark.write('mlp_784_10\t')
        bmark.write('pybrain_arac atlas=OS_version\t')
        bmark.write('%.2f\n'%t)
        bmark.write('# Pybrain does not support NLL, squared error used \n')
        bmark.write('# Pybrain does not support minibatches \n')

        del fnn
        del trainer


        fnn = buildNetwork(784,500,10, outclass = SoftmaxLayer, fast = True)
        trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
        t = time.time()
        trainer.trainEpochs(1)
        t = time.time() - t
        bmark.write('mlp_784_500_10\t')
        bmark.write('pybrain_arac atlas=OS_version\t')
        bmark.write('%.2f\n'%t)
        bmark.write('# Pybrain does not support NLL, squared error used \n')
        bmark.write('# Pybrain does not support minibatches \n')


        del fnn
        del trainer


        fnn = buildNetwork(784,1000,1000,1000,10, outclass = SoftmaxLayer, fast = True)
        trainer = BackpropTrainer(fnn, dataset = data784, learningrate = 0.01, verbose = True)
        t = time.time()
        trainer.trainEpochs(1)
        t = time.time() - t
        bmark.write('mlp_784_1000__1000_1000_10\t')
        bmark.write('pybrain_arac atlas=OS_version\t')
        bmark.write('%.2f\n'%t)
        bmark.write('# Pybrain does not support NLL, squared error used \n')
        bmark.write('# Pybrain does not support minibatches \n')
        
        bmark.close()

if __name__ == '__main__':
    benchmarking(arac = True)
