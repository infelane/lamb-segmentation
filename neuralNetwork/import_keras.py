import tensorflow as tf
if int(tf.__version__.split('.')[0]) >= 2:
    from tensorflow.keras.models import Model
    from tensorflow.layers import Conv2D, Input, Flatten, Dense
    from tensorflow.losses import categorical_crossentropy
    from tensorflow.optimizers import Nadam, Adam, SGD

    from tensorflow.keras.callbacks import LearningRateScheduler

    from tensorflow.keras.preprocessing.image import ImageDataGenerator as ImageDataGeneratorOrig, NumpyArrayIterator, \
        array_to_img
    
    """ NumpyArrayIterator is inherited from NumpyArrayIteratorPre """
    from keras.preprocessing.image import image
    NumpyArrayIteratorPre = image.NumpyArrayIterator

    from tensorflow.keras import backend as K
    
else:
    from keras.models import Model
    from keras.layers import Conv2D, Input, Flatten, Dense
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Nadam, Adam, SGD

    from keras.callbacks import LearningRateScheduler

    from keras.preprocessing.image import ImageDataGenerator as ImageDataGeneratorOrig, NumpyArrayIterator, array_to_img

    """ NumpyArrayIterator is inherited from NumpyArrayIteratorPre"""
    from keras.preprocessing.image import image
    NumpyArrayIteratorPre = image.NumpyArrayIterator
    
    from keras import backend as K
    