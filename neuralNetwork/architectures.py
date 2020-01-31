# from neuralNetwork.import_keras.keras_general.layers import Input, Conv2D, BatchNormalization, Activation
# from neuralNetwork.import_keras.keras_general.models import Model

from neuralNetwork.import_keras import Input, Conv2D, BatchNormalization, Activation, Model


def fullyConnected1x1(n_in, k=1, batch_norm=False):
    """
    
    :param n_in:
    :param k:
    :param batch_norm: If True, is applied after activation  https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
    :return:
    """

    # BN before activation (although originally), seems worse in literature
    batch_norm_before_act = False
    
    shape = (None, None, n_in)
    
    inputs = Input(shape=shape)
    
    if batch_norm & batch_norm_before_act:
        l = Conv2D(k, (1, 1))(inputs)
        l = BatchNormalization()(l)
        l = Activation(activation='elu')(l)
    else:
        l = Conv2D(k, (1, 1), activation='elu')(inputs)

        if batch_norm & ~batch_norm_before_act:
            l = BatchNormalization()(l)

    outputs = Conv2D(2, (1, 1), activation='softmax')(l)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def convNet(n_in, k=1, batch_norm=False):
    """

    # :param n_in:
    # :param k:
    # :param batch_norm: If True, is applied after activation  https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
    # :return:
    """

    # BN before activation (although originally), seems worse in literature

    shape = (None, None, n_in)

    inputs = Input(shape=shape)

    # TODO padding not *same*
    l = Conv2D(k, (3, 3), activation='elu', padding='same')(inputs)

    if batch_norm:
        l = BatchNormalization()(l)

    outputs = Conv2D(2, (1, 1), activation='softmax')(l)

    model = Model(inputs=inputs, outputs=outputs)

    return model