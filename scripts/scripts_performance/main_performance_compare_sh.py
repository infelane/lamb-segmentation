import matplotlib.pyplot as plt

import tensorflow as tf

from methods.basic import NeuralNet
from plotting import concurrent

"""
Let's check if it can work with eager exe. Not exactly
"""
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


class Main(object):
    def __init__(self):

        # data
        self.data()

        # Load model(s)

        model_name = 'unet'   # ['simple', 'ti-unet', 'unet']:

        folder = f'C:/Users/admin/Data/ghent_altar/net_weight/{model_name}_d1_k9_n80'
        epoch = 1
        path = f'C:/Users/admin/Data/ghent_altar/net_weight/{model_name}_d1_k9_n80/w_{epoch}.h5'

        from scripts.scripts_performance.main_performance import load_model_quick
        model = load_model_quick(path)
        neural_net = NeuralNet(model, w_ext=10, norm_x=True)

        model.summary()

        for epoch in range(1, 10+1):
            print('epoch', epoch)
            neural_net.load(folder, epoch)

            # Predict
            y_pred = neural_net.predict(self.img_x)

            if 0:
                plt.imshow(y_pred[..., 0])
                plt.show()



            for val_name in self.val:
                print(val_name)

                y_true_val = self.val[val_name]

                data_i = _eval_func_single(y_true_val, y_pred)

                print(data_i)

        # TODO best performing (ti-unet: 4)
        neural_net.load(folder, 4)

        y_pred = neural_net.predict(self.img_x)

        from performance.testing import get_y_pred_thresh
        y_pred_thresh = get_y_pred_thresh(y_pred, data_i['thresh'])

        concurrent([self.img_x[..., :3], self.img_y[..., 0], y_pred[..., 0], y_pred_thresh[..., 0]])

        y_pred

    def data(self):
        from datasets.default_trainingsets import xy_from_df, get_13zach, panel13withoutRightBot
        img_x, img_y = xy_from_df(get_13zach(), 5)

        self.img_x, self.img_y = img_x, img_y

        _, img_y_2 = panel13withoutRightBot(img_y)

        # Validation data
        self.val = {'13_botright_all': img_y_2}


if __name__ == '__main__':

    Main()
