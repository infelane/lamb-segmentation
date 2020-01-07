from .basic import NeuralNet

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Nadam

from data.modalities import modality_exist


def architecture0(n_in, k=1):
    shape = (None, None, n_in)
    
    inputs = Input(shape=shape)
    l = Conv2D(k, (1, 1), activation='elu')(inputs)
    outputs = Conv2D(2, (1, 1), activation='softmax')(l)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def compile0(model, lr=1e5):
    model.compile(Nadam(lr), loss=categorical_crossentropy)


def neuralNet0(mod):
    
    modality_exist(mod)
    n_in = 12 if mod == 'all' else 3 if mod == 'clean' else NotImplementedError()

    model = architecture0(n_in, k=20)
    compile0(model)

    n = NeuralNet(model)

    return n
