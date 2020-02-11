import matplotlib.pyplot as plt
import pandas as pd

from .import_keras import *


def find_learning_rate(model, training_data, class_weight, verbose=1):
    
    lr0 = 1e-5
    lr1 = 1e+5
    n = 20

    lr_range = [lr0 * (lr1/lr0)**(i/n) for i in range(n+1)]

    def scheduler(epoch):
        assert epoch <= n
        return lr_range[epoch]

    lrScheduler = LearningRateScheduler(scheduler)

    from keras.preprocessing.image import Iterator
    if isinstance(training_data, Iterator):
        flow_tr = training_data     # alias
        hist = model.fit_generator(flow_tr, epochs=n+1, steps_per_epoch=100, callbacks=[lrScheduler],
                                   class_weight=class_weight,
                                   verbose=verbose)
    else:
        hist = model.fit(*training_data, epochs=n+1, callbacks=[lrScheduler], class_weight=class_weight,
                         verbose=verbose)
    
    df = pd.DataFrame.from_dict(hist.history)
    
    lr_min = df['lr'][df['loss'].idxmin()]

    lr_optimal = 0.1*lr_min
    
    print_string = f'Expected optimal learning rate: {lr_optimal}'
    print(print_string)

    if verbose:
        df.plot('lr', 'loss', logx=True)
        plt.title(print_string)
        
    return lr_optimal
    