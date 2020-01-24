from .basic import NeuralNet

from neuralNetwork.import_keras import SGD, categorical_crossentropy
from neuralNetwork.architectures import fullyConnected1x1
from data.modalities import modality_exist

from performance.metrics import accuracy_with0, jaccard_with0


def compile0(model, lr=1e-1):
    
    # optimizer = Adam(lr)
    # optimizer = Nadam(lr)

    optimizer = SGD(lr)

    metrics = [accuracy_with0, jaccard_with0]
    
    model.compile(optimizer, loss=categorical_crossentropy, metrics=metrics)


def neuralNet0(mod, lr=None):
    
    modality_exist(mod)
    n_in = 12 if mod == 'all' else 3 if mod == 'clean' else NotImplementedError()

    model = fullyConnected1x1(n_in, k=20, batch_norm=True)
    
    model.summary()
    
    args = (lr, ) if (lr is not None) else ()
    compile0(model, *args)

    n = NeuralNet(model)

    return n
