import numpy as np

try:
    import tensorflow.keras.backend as K
except (ImportError, ModuleNotFoundError):
    import keras.backend as K

# import tensorflow.keras.backend as K
import tensorflow as tf

from .basic import NeuralNet

try:
    from tensorflow.keras.losses import mse, binary_crossentropy
except (ImportError, ModuleNotFoundError):
    from keras.losses import mse, binary_crossentropy

from neuralNetwork.import_keras import SGD, categorical_crossentropy, Adam, Nadam
from neuralNetwork.architectures import fullyConnected1x1, convNet, unet, ti_unet, autoencoder
from data.modalities import get_mod_n_in

from performance.metrics import accuracy_with0, jaccard_with0


def compile_segm(model, lr=1e-4, class_weights=(1, 1)):
    
    # optimizer = SGD(lr)
    # optimizer = Adam(lr)
    optimizer = Nadam(lr)

    metrics = [accuracy_with0, jaccard_with0, kappa_loss]

    if 0: loss = categorical_crossentropy
    else: loss = weighted_categorical_crossentropy(class_weights)
    
    model.compile(optimizer, loss=loss, metrics=metrics)
    
    
def compile_ae(model, lr=0.002):
    optimizer = Nadam(lr)

    metrics = [mse]
    
    loss = binary_crossentropy  # THIS SHOULD NOT HAVE WORKED! categorical_crossentropy
    
    model.compile(optimizer, loss=loss, metrics=metrics)


def neuralNet0(mod, lr=None, k=20, verbose=1, class_weights=None,
               batch_norm=True
               ):

    n_in = get_mod_n_in(mod)
    
    if 0:
        model = fullyConnected1x1(n_in, k=k, batch_norm=batch_norm)
        w_ext = 0
    elif 0:
        model = convNet(n_in, k=k, batch_norm=batch_norm)
        w_ext = 2
    elif 0:
        model = unet(n_in, filters=k, batch_norm=batch_norm)
        w_ext = 2
    else:
        w_ext = 10
        model = ti_unet(n_in, filters=k, w=10, ext_in=w_ext//2, batch_norm=batch_norm)
 
    if verbose:
        model.summary()
    
    args = {}
    if lr is not None: args['lr'] = lr
    if class_weights is not None: args['class_weights'] = class_weights
    
    compile_segm(model, **args)

    n = NeuralNet(model, w_ext=w_ext)

    return n


def get_neural_net_ae(mod, k=None, lr=None, w_in=None, depth=2, b_double=True, b_split_mod=False,
                      batch_norm = False,
                      verbose=1):
    
    n_in = get_mod_n_in(mod)
    
    model = autoencoder(n_in, k, w_in=w_in, depth=depth, b_double=b_double, b_split_modality=b_split_mod,
                        batch_norm = batch_norm)
    
    if verbose:
        model.summary()
    
    args = {}
    if lr is not None:
        args['lr'] = lr
    compile_ae(model, **args)
    
    w_ext = 28 if depth == 2 else 12 if depth == 1 else -1  # To implement
    n = NeuralNet(model, w_ext=w_ext)
    
    return n


def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """

    if not isinstance(weights, tf.Variable):
        weights = K.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.math.log(output) * weights
            return - tf.reduce_sum(weighted_losses, len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')

    return loss


def kappa_loss(y_pred, y_true, y_pow=1, eps=1e-10, N=2, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    y_pred = tf.reshape(y_pred, (-1, N))
    y_true = tf.reshape(y_true, (-1, N))

    with tf.name_scope(name):
        y_true = tf.cast(y_true, float)
        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), float)
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, float)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.cast(bsize, float))
    
        return nom / (denom + eps)
