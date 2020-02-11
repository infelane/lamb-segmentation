# from neuralNetwork.import_keras.keras_general.layers import Input, Conv2D, BatchNormalization, Activation
# from neuralNetwork.import_keras.keras_general.models import Model

from neuralNetwork.import_keras import Input, Conv2D, BatchNormalization, Activation, Model
from keras.layers import Cropping2D, Concatenate, Dropout, MaxPooling2D
# from tensorflow.keras.layers import Cropping2D, Concatenate, Dropout, MaxPooling2D

def fullyConnected1x1(n_in, k=1, w_in=None, batch_norm=False):
    """
    
    :param n_in:
    :param k:
    :param w_in: None if you want it to be usable for all input widths
    :param batch_norm: If True, is applied after activation  https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
    :return:
    """

    # BN before activation (although originally), seems worse in literature
    batch_norm_before_act = False
    
    shape = (w_in, w_in, n_in)
    
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


def convNet(n_in, k=1, w_in=None, batch_norm=False, padding='valid'):
    """

    :param n_in:
    :param k:
    :param w_in: None if you want it to be usable for all input widths
    :param batch_norm: If True, is applied after activation  https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
    :return:
    """

    # BN before activation (although originally), seems worse in literature

    shape = (w_in, w_in, n_in)

    inputs = Input(shape=shape)

    l = Conv2D(k, (3, 3), activation='elu', padding=padding)(inputs)

    if batch_norm:
        l = BatchNormalization()(l)

    outputs = Conv2D(2, (1, 1), activation='softmax')(l)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def unet(features_in, w=10, ext_in=0, filters=2, max_depth=1):
    """
    Builds up the original U-Net
    :param features_in:
    :param w:
    :param filters:
    :param max_depth: amount of downsamples original U-Net 4
        if max_depth 1, w_in%2==0
        if max_depth 2, w_in%4==0
        if max_depth 3, (w_in+4)%8==0
        if max_depth 4, (w_in+4)%16==0

        (if max_depth 0: w_out - w_in = 4)
        if max_depth 1: w_out - w_in = 16
        if max_depth 2: w_out - w_in = 40
        if max_depth 3: w_out - w_in = 88
        if max_depth 4: w_out - w_in = 184
        # Everytime *2 + 8
    :return:
    """
    
    diff_in_out = [4, 16, 40, 88, 184]
    
    w_in_rest = (w + diff_in_out[max_depth] + 4) % (2 ** max_depth)
    w_in_perf = w + diff_in_out[max_depth] + w_in_rest
    w_out_perf = w + w_in_rest
    
    assert w_in_perf <= w + 2 * ext_in, 'ext_in is too small'
    
    act = 'elu'
    
    w_in = w + ext_in * 2
    input_shape = [w_in, w_in, features_in]
    inputs = Input(shape=input_shape)
    
    def left_block(depth):
        """
        :param depth: starts at 0, +1 after each downpooling
        :return: a function that has to be initalized by giving previous layer
        """
        
        def foo(l_prev):
            lefti = _gen_conv(filters * (2 ** depth), act=act, name='left{}_0'.format(depth))(l_prev)
            lefti = _gen_conv(filters * (2 ** depth), act=act, name='left{}_1'.format(depth))(lefti)
            return lefti
        
        return foo
    
    def right_block(depth):
        """
        :param depth:
        :return:
        """
        
        def foo(l_prev):
            righti = _gen_conv(filters * (2 ** depth), act=act, name='right{}_0'.format(depth))(l_prev)
            righti = _gen_conv(filters * (2 ** depth), act=act, name='right{}_1'.format(depth))(righti)
            return righti
        
        return foo
    
    delta_w = w_in - w_in_perf
    ext_l = delta_w // 2
    ext_r = delta_w - ext_l
    input_crop = Cropping2D(((ext_l, ext_r), (ext_l, ext_r)))(inputs)
    
    left_lst = []
    w_left_lst = []
    for i in range(max_depth + 1):
        if i == 0:
            downi = input_crop
            # TODO
            # w_left_lst.append(w_in-4)
            w_left_lst.append(w_in_perf - 4)
        else:
            downi = MaxPooling2D((2, 2), strides=(2, 2), name='down{}'.format(i - 1))(left_lst[-1])
            w_left_lst.append(w_left_lst[-1] // 2 - 4)
        lefti = left_block(i)(downi)
        left_lst.append(lefti)
    
    right_lst = [None for _ in range(max_depth)]
    w_right_lst = [None for _ in range(max_depth)]
    for i in range(max_depth - 1, -1, -1):  # starts at max_depth-1!
        if i == max_depth - 1:
            upi = Conv2DTranspose(filters=filters * (2 ** i), kernel_size=(2, 2), strides=(2, 2),
                                  name='up{}'.format(i))(left_lst[i + 1])
            w_right_lst[i] = w_left_lst[i + 1] * 2
        else:
            upi = Conv2DTranspose(filters=filters * (2 ** i), kernel_size=(2, 2), strides=(2, 2),
                                  name='up{}'.format(i)
                                  )(right_lst[i + 1])
            w_right_lst[i] = (w_right_lst[i + 1] - 4) * 2
        
        delta_w = w_left_lst[i] - w_right_lst[i]
        ext_l = delta_w // 2
        ext_r = delta_w - ext_l
        
        left_crop = Cropping2D(((ext_l, ext_r), (ext_l, ext_r)))(left_lst[i])
        
        righti = Concatenate()([left_crop, upi])
        righti = right_block(i)(righti)
        right_lst[i] = righti
    
    outputs = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', name='fcc')(right_lst[0])
    # outputs = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax')(right0)
    
    # delta_w = w_right_lst[0] - 4 - w
    delta_w = w_out_perf - w  # TODO
    ext_l = delta_w // 2
    ext_r = delta_w - ext_l
    outputs = Cropping2D(((ext_l, ext_r), (ext_l, ext_r)))(outputs)
    
    model = Model(inputs, outputs)
    return model


def ti_unet(features_in, w=10, ext_in=0, filters=2, max_depth=1, dropout=False, double=False,
            n_per_block=1,
            batch_norm=False):
    """
    Builds up the original U-Net that is Translation Equivariant
    :param features_in:
    :param w:
    :param filters:
        (if max_depth 0: w_out - w_in = 4)  (vs 4)
        if max_depth 1: w_out - w_in = 18   (vs 16)
        if max_depth 2: w_out - w_in = 46   (vs 40)
        if max_depth 3: w_out - w_in = 102  (vs 88)
        if max_depth 4: w_out - w_in = 214  (vs 184)
        Everytime *2 + 10
    :param double: double the amount of filters after each pooling
    :param double: amount of convolutional layers per block (2 original, 1 for reduced)
    :return: the model of the TE-U-Net
    """
    
    diff_in_out = [4, 18, 46, 102, 214] if n_per_block == 2\
        else [None, 10, None, None, None] if n_per_block == 1\
        else NotImplementedError(n_per_block)
    
    w_in_perf = w + diff_in_out[max_depth]
    
    assert w_in_perf <= w + 2 * ext_in, 'ext_in is too small'
    
    act = 'elu'
    
    w_in = w + ext_in * 2
    input_shape = [w_in, w_in, features_in]
    inputs = Input(shape=input_shape)
    
    def left_block(depth):
        """
        :param depth: starts at 0, +1 after each downpooling
        :return: a function that has to be initalized by giving previous layer
        """
        dr = 2 ** depth
        
        def foo(l_prev):
            f = filters * (2 ** depth) if double else filters

            lefti = l_prev
            for i_per_block in range(n_per_block):
                lefti = _gen_conv(f, act=act, dr=dr, name=f'left{depth}_{i_per_block}')(lefti)
                
                if batch_norm:
                    lefti = BatchNormalization(name=f'batchnorm_left{depth}')(lefti)
                
            return lefti
        
        return foo
    
    def right_block(depth):
        """
        :param depth:
        :return:
        """
        dr = 2 ** depth
        
        def foo(l_prev):
            f = filters * (2 ** depth) if double else filters
            
            righti = l_prev
            for i_per_block in range(n_per_block):
                righti = _gen_conv(f, act=act, dr=dr, name=f'right{depth}_{i_per_block}')(righti)
                if batch_norm:
                    righti = BatchNormalization(name=f'batchnorm_right{depth}')(righti)
            return righti
        
        return foo
    
    delta_w = w_in - w_in_perf
    
    ext_l = delta_w // 2
    ext_r = delta_w - ext_l
    input_crop = Cropping2D(((ext_l, ext_r), (ext_l, ext_r)))(inputs)
    
    left_lst = []
    w_left_lst = []
    for i in range(max_depth + 1):
        if i == 0:
            downi = input_crop
            
        else:
            pool_w = 1 * 2 ** (i - 1) + 1
            downi = MaxPooling2D((pool_w, pool_w), strides=(1, 1), name='down{}'.format(i - 1))(left_lst[-1])
    
        lefti = left_block(i)(downi)
        left_lst.append(lefti)
        
        # Get width of layer
        if 0:
            # Complicated
            
            if i == 0:
                w_i = w_in_perf - n_per_block * (2 * (2 ** i) + 1 - 1)
            else:
                w_i = w_left_lst[-1] - (pool_w - 1) - n_per_block * (2 * (2 ** i) + 1 - 1)
        else:
            w_i = lefti._shape_tuple()[-2]
        w_left_lst.append(w_i)
    
    right_lst = [None for _ in range(max_depth)]
    w_right_lst = [None for _ in range(max_depth)]
    for i in range(max_depth - 1, -1, -1):  # starts at max_depth-1!
        f = filters * (2 ** i) if double else filters
        li = Conv2D(filters=f, kernel_size=(2, 2), dilation_rate=2 ** i,
                    name='up{}'.format(i)
                    )
        if i == max_depth - 1:
            upi = li(left_lst[i + 1])
        
        else:
            upi = li(right_lst[i + 1])
        if batch_norm:
            upi = BatchNormalization(name=f'batchnorm_up{i}')(upi)
            
        if 0:
            if i == max_depth - 1:
                w_right_i = w_left_lst[i + 1] - (2 ** i)
            else:
                w_right_i = w_right_lst[i + 1] - n_per_block * (2 * (2 ** (i + 1)) + 1 - 1) - (2 ** i)
        else:
            w_right_i = upi._shape_tuple()[-2]
        w_right_lst[i] = w_right_i
        
        delta_w = w_left_lst[i] - w_right_lst[i]
        ext_l = delta_w // 2
        ext_r = delta_w - ext_l
        
        left_crop = Cropping2D(((ext_l, ext_r), (ext_l, ext_r)))(left_lst[i])
        
        righti = Concatenate()([left_crop, upi])
        righti = right_block(i)(righti)
        right_lst[i] = righti
    
    l_last = right_lst[0]
    if dropout:
        # 0.1 should be low
        l_last = Dropout(rate=0.1, noise_shape=(tf.shape(right_lst[0])[0], w, w, filters))(right_lst[0])
        
    outputs = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', name='fcc')(l_last)
    
    assert outputs._shape_tuple()[-3] == outputs._shape_tuple()[-2], outputs._shape_tuple()
    w_out_diff = w - outputs._shape_tuple()[-2]
    assert w_out_diff == 0, f'diff_in_out should be {diff_in_out[max_depth] + w_out_diff} instead of {diff_in_out[max_depth]}'
    
    model = Model(inputs, outputs)
    return model


def _conv_batch_norm(filters,
                     kernel_size,
                     strides=(1, 1),
                     dilation_rate=(1, 1),
                     name=None,
                     activation=None):
    def foo(bar):
        l1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, name=name)(
            bar)
        l2 = BatchNormalization()(l1)
        l3 = Activation(activation=activation)(l2)
        return l3
    
    return foo


def _gen_conv(filters, act='elu', name=None, dr=1, batch_norm=True):
    """
    
    :param filters:
    :param act:
    :param name:
    :param dr:
    :param batch_norm: Batchnorm before activation. Not necessarily useful.
    :return:
    """
    if batch_norm:
        return _conv_batch_norm(filters=filters, kernel_size=(3, 3), dilation_rate=(dr, dr), name = name, activation=act)
    else:
        return Conv2D(filters=filters, kernel_size=(3, 3), dilation_rate=(dr, dr), activation=act,
                      name=name
                      )