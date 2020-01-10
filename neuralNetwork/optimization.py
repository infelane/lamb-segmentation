import matplotlib.pyplot as plt
import pandas as pd

from .import_keras import *


def find_learning_rate(model, training_data):
    
    lr0 = 1e-10
    lr1 = 1e+10
    n = 20

    lr_range = [lr0 * (lr1/lr0)**(i/n) for i in range(n+1)]

    def scheduler(epoch):
        assert epoch <= n
        return lr_range[epoch]

    lrScheduler = LearningRateScheduler(scheduler)
    
    hist = model.fit(*training_data, epochs=n+1, callbacks=[lrScheduler])
    
    df = pd.DataFrame.from_dict(hist.history)
    
    lr_min = df['lr'][df['loss'].idxmin()]

    lr_optimal = 0.1*lr_min
    
    print_string = f'Expected optimal learning rate: {lr_optimal}'
    print(print_string)

    plt.figure()
    df.plot('lr', 'loss', logx=True)
    plt.title(print_string)
    plt.show()
    
    return lr_optimal
    