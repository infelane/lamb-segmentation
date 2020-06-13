
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/admin/OneDrive - ugentbe/data/dataframes/transfer_learning_lamb4.csv', delimiter=';')

    # Remove possible duplicates!
    # df.drop_duplicates()

    print(df[df.duplicated(['epoch', 'dataset'])])
    df = df.drop_duplicates(['epoch', 'dataset'], keep='last')

    def plot(df, metric):
        fig, ax = plt.subplots()
        for label, df_i in df.groupby('dataset'):
            df_i.sort_values('epoch').plot('epoch', metric, ax=ax, label=label)
        plt.legend()
        plt.ylabel(metric)
        plt.show()

    if 0:
        plot(df, 'kappa')
        plot(df, 'jaccard')

    # Smoother plot?

    def plot_smooth(df, metric):

        fig, ax = plt.subplots()
        for label, df_i in df.groupby('dataset'):

            xy = df_i.sort_values('epoch')

            x = xy['epoch']
            y = xy[metric]

            if 0:
                # Very simple average filter

                l = 5
                v = np.ones(l) / l
                yhat = np.convolve(y, v, 'same')

            else:
                from scipy.signal import savgol_filter
                l = 21
                yhat = savgol_filter(y, l, 3)  # window size 51, polynomial order 3

            a = plt.plot(x, y, 'x', label=label)
            plt.plot(x, yhat, '-', c=a[0].get_color())

        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.show()

    # plot_smooth(df, 'jaccard')
    plot_smooth(df, 'kappa')

    import tikzplotlib
    tikzplotlib.save("C:/Users/admin/OneDrive - ugentbe/data/dataframes/transfer_learning.tikz")

    # 300 represents number of points to make between T.min and T.max

    spl = make_interp_spline(T, power, k=3)  # type: BSpline
    power_smooth = spl(xnew)

    plt.plot(xnew, power_smooth)
    plt.show()

    print("Done")
