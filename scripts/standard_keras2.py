import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Flexible GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
