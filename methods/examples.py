from .basic import NeuralNet

from neuralNetwork.import_keras import SGD, categorical_crossentropy
from neuralNetwork.architectures import fullyConnected1x1, convNet
from data.modalities import _modality_exist

from performance.metrics import accuracy_with0, jaccard_with0


def compile0(model, lr=1e-1):
    
    # optimizer = Adam(lr)
    # optimizer = Nadam(lr)

    optimizer = SGD(lr)

    metrics = [accuracy_with0, jaccard_with0]
    
    model.compile(optimizer, loss=categorical_crossentropy, metrics=metrics)


def neuralNet0(mod, lr=None):
    
    _modality_exist(mod)
    if mod == 'all':
        n_in = 12
    elif mod == 'clean':
        n_in = 3
    else:
        try:
            if int(mod) == 5:
                n_in = 9
            else: NotImplementedError()
        except ValueError as verr:
            pass
            NotImplementedError()
    
    k = 20
    batch_norm = False
    if 0:
        model = fullyConnected1x1(n_in, k=k, batch_norm=batch_norm)
    else:
        model = convNet(n_in, k=k, batch_norm=batch_norm)
    
    model.summary()
    
    args = (lr, ) if (lr is not None) else ()
    compile0(model, *args)

    n = NeuralNet(model)

    return n
