import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Main(object):
    def __init__(self):
        folder = '/home/lameeus/data/ghent_altar/dataframes/'
        file = 'pretrained_unet_10lamb_kfold_single.csv'
        
        file2 = 'tiunet_10lamb_kfold_single.csv'
        
        df_unet_pretrained_fixed = pd.read_csv(os.path.join(folder, file), sep=';')
        df_tiunet_single = pd.read_csv(os.path.join(folder, file2), sep=',')
        df_tiunet_single = remove_duplicates(df_tiunet_single)

        df_tiunet_avg = pd.read_csv(os.path.join(folder, 'tiunet_10lamb_kfold_avgpred.csv'), sep=',')
        df_tiunet_avg = remove_duplicates(df_tiunet_avg)

        if 0:
            self.correlation_jaccard_kappa(df_tiunet_single)
        
        # Very nice graph to illustrate performance in terms of k and epoch
        if 0:
            # ['median', 'mean']. Median is best
            for mode in ['median']:
                # ['jaccard', 'kappa']  # Both OK, but jaccard is easier to understand
                for y in ['jaccard']:
                    for x in ['k', 'epoch']:
                        self.plot_shaded_line(df_tiunet_single, x=x, y=y, mode=mode)
                        
                        self.plot_shaded_line(df_tiunet_single, x=x, y=y, mode=mode, b_folds=False)
                 
            self.plot2d(df_tiunet_single)
        else:
            # ['median', 'mean']. Median is best
            for mode in ['median']:
                # ['jaccard', 'kappa']  # Both OK, but jaccard is easier to understand
                for y in ['jaccard']:
                    for x in ['k', 'epoch_start']:
                        self.plot_shaded_line(df_tiunet_avg, x=x, y=y, mode=mode)
                        self.plot_shaded_line(df_tiunet_avg, x=x, y=y, mode=mode, b_folds=False)

        if 1:
            print('TI-UNet single')
            k_start_range = range(1, 30+1)
            a_lst = []
            a_mean = []
            for k_start in k_start_range:
                a = self.estimated_performance(df_tiunet_single, k_start=k_start,
                                               verbose=0)
                a_lst.append(a)
                a_mean.append(np.mean(a))
            plt.figure()
            if 1:
                for i_fold in range(6):
                    plt.plot(k_start_range, list(zip(*a_lst))[i_fold], label=i_fold)
            plt.plot(k_start_range, a_mean, label='mean')
            plt.legend()
            plt.xlabel('k')
            plt.show()
            
            epoch_start_range = range(1, 40+1)
            a_lst = []
            a_mean = []
            for epoch_start in epoch_start_range:
                a = self.estimated_performance(df_tiunet_single, epoch_start=epoch_start,
                                               verbose=0)
                a_lst.append(a)
                a_mean.append(np.mean(a))
            plt.figure()
            if 1:
                for i_fold in range(6):
                    plt.plot(epoch_start_range, list(zip(*a_lst))[i_fold], label=i_fold)
            plt.plot(epoch_start_range, a_mean, label='mean')
            plt.legend()
            plt.xlabel('epoch start')
            plt.show()
            
            """
            For TI-UNet we use
            k_start=20,
            epoch_start=20
            """
            k_start = 20
            epoch_start = 20
            self.estimated_performance(df_tiunet_single,
                                       k_start = k_start,
                                       epoch_start=epoch_start,
                                       verbose=1
                                       )
            
            """
            For TI-UNet avg pred we use
            k_start=15,
            epoch_start=25
            """
            print('TI-UNet avg pred')
            k_start=15,
            epoch_start=25
            self.estimated_performance(df_tiunet_avg,
                                       k_start = k_start,
                                       epoch_start=epoch_start,
                                       verbose=1
                                       )
            
            
    def correlation_jaccard_kappa(self, df):
        plt.figure()
        for i_fold in range(6):
            df_i = df[df.i_fold == i_fold]
            
            plt.subplot(3, 2, i_fold + 1)
            plt.hist2d(df_i['kappa'], df_i['jaccard'], bins=20)
            plt.title(f'i_fold = {i_fold}')
            plt.xlabel('kappa')
            plt.ylabel('jaccard')
        plt.show()

    def plot_shaded_line(self, df, x='k', y='jaccard', mode: ['mean', 'median'] = 'mean', b_folds = True):

        fig, ax = plt.subplots()
        
        def sub_method(label=None):
            if mode == 'mean':
                y_mean = m_groupx[y].mean()
                y_std = m_groupx[y].std()
                y_lower = y_mean - y_std
                y_upper = y_mean + y_std
            elif mode == 'median':
                y_mean = m_groupx[y].median()
                y_lower = m_groupx[y].quantile(0.25)
                y_upper = m_groupx[y].quantile(0.75)

            plt.ylabel(y)
            y_mean.plot(ax=ax, label=label)
            ax.fill_between(y_mean.keys(), y_lower, y_upper, alpha=0.35)

        if b_folds:
    
            for i_fold, m in df.groupby('i_fold'):
                m_groupx = m.groupby(x)

                sub_method(label=f'i_fold = {i_fold}')
            
        else:
            # Take mean of mode per i_fold
            m_groupx = df.groupby(x)

            a = df.groupby([x, 'i_fold'])
            if mode == 'mean':
                y_mean = a[y].mean()
                y_std = a[y].std()
                y_lower = y_mean - y_std
                y_upper = y_mean + y_std
            elif mode == 'median':
                y_mean = a[y].median()
                y_lower = a[y].quantile(0.25)
                y_upper = a[y].quantile(0.75)

            def avg_per_i_fold(y_groupby):
                return y_groupby.reset_index().groupby(x)[y].mean()

            plt.ylabel(y)
            avg_per_i_fold(y_mean).plot(ax=ax)
            ax.fill_between(avg_per_i_fold(y_mean).keys(), avg_per_i_fold(y_lower), avg_per_i_fold(y_upper), alpha=0.35)
    
        plt.legend()
        if mode == 'mean':
            plt.title('mean and std')
        elif mode == 'median':
            plt.title('median and Q1, Q3')
        plt.show()

    def plot2d(self, df, x='k', y='epoch', z='jaccard'):

        fig, ax = plt.subplots()
        for i_fold in range(6):
            plt.subplot(3, 2, i_fold + 1)
            a = np.zeros(shape=(df[x].max() - df[x].min() + 1, df[y].max() - df[y].min() + 1))
            for i in range(30):
                for j in range(40):
                    a[i, j] = df[(df[x] == i + 1) & (df[y] == j + 1) & (df['i_fold'] == i_fold)][z]
            plt.imshow(a, cmap=plt.cm.jet, aspect='auto', origin='lower', alpha=1, interpolation='none',
                       extent=(df[x].min(), df[x].max(), df[y].min(), df[y].max())
                       )
            plt.title(f'i_fold = {i_fold}')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.colorbar().set_label(z)
        plt.show()

    def estimated_performance(self, df, metric='jaccard',
                              epoch_start = 21,
                              k_start = 10,
                              verbose=1):
        
        filter1 = (df.epoch >= epoch_start) if 'epoch' in df.keys() else\
            (df.epoch_start == epoch_start)
        df_sub = df[filter1 & (df.k >= k_start)]
        
        perf_fold = df_sub.groupby('i_fold')[metric].median()
        
        if verbose:
            # Median Performance per i_fold
            for i_fold in range(6):
                print(f'i_fold: {i_fold}, {metric} = {perf_fold[i_fold]}')

        if verbose:
            # Average median performance
            print(f'Average {metric} = {perf_fold.mean()}')
        
        return perf_fold
        
        
def remove_duplicates(df):
    
    # df_unique = df.drop_duplicates(subset=['k', 'i_fold', 'epoch'], keep='last')
    df_unique = df.drop_duplicates()
    
    return df_unique


def main():
    
    folder = '/home/lameeus/data/ghent_altar/dataframes/'
    file = 'pretrained_unet_10lamb_kfold_single.csv'
    
    file2 = 'tiunet_10lamb_kfold_single.csv'
    
    df_unet_pretrained_fixed = pd.read_csv(os.path.join(folder, file), sep=';')
    df_tiunet_single = pd.read_csv(os.path.join(folder, file2), sep=',')
    df_tiunet_single = remove_duplicates(df_tiunet_single)
    
    if 0:
        def get0(df):
            return df.groupby(['k', 'i_fold'], as_index=False).mean()
        
        a = get0(df_unet_pretrained_fixed)
        b = get0(df_tiunet_single)
        
        i_fold = 0
        
        a[a['i_fold']==i_fold].plot('k', 'kappa',  style='*')
    
        # plot data
        fig, ax = plt.subplots()
        # use unstack()
        df_unet_pretrained_fixed.groupby(['k', 'i_fold']).mean()['kappa'].unstack().plot(ax=ax, style='*')
    
        for i_fold in range(6):
            if i_fold == 0:
                ax = None
            ax = b[b['i_fold']==i_fold].plot('k', 'kappa',  style='-*', ax=ax, label=f'i_fold = {i_fold}')
        plt.legend()
        plt.show()
    
    # 3D plot
    if 0:

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        
        c = df_tiunet_single[df_tiunet_single['i_fold']==0]
        ax.plot_trisurf(c.k, c.epoch, c.kappa,
                        # cmap=cm.jet,
                        # linewidth=0.2
                        )
        plt.show()
        
    def plot1(df, x, y='kappa', average=None, filter=None):
        
        df_filter = df[filter] if filter is not None else df
        
        # plt.figure()
        
        if 1:
            for i_fold in range(6):
                df[df.i_fold == i_fold].hist(column=['kappa', 'jaccard'], bins=20)
        else:
            df.hist(column=['kappa', 'jaccard'], bins=20)
        
        if 1:
                
            df_filter_group =  df_filter.groupby(x)
            
            df_jaccard = df_filter_group['jaccard']
            
            y_mean =df_jaccard.mean()
            y_std = df_jaccard.std()
            
            fig, ax = plt.subplots()
            plt.ylabel('jaccard')
            y_mean.plot(ax=ax)
            ax.fill_between(y_mean.keys(), y_mean - y_std, y_mean + y_std, alpha=0.35)

            
        df_filter.plot(x, y, style='-x')
        
    lst_y = 'kappa', 'jaccard'
    plot1(df_tiunet_single, 'k', average=df_tiunet_single['i_fold'] == 0)
    
    """
    Average
    """

    # TODO add errorbar/errorfill
    # TODO plot simultaniously
    plt.figure()
    plt.subplot(2, 1, 1)
    df_tiunet_single[df_tiunet_single.i_fold==0].groupby('k', as_index=False).mean().plot('k', 'kappa')
    plt.subplot(2, 1, 2)
    df_tiunet_single[df_tiunet_single.i_fold==0].groupby('epoch', as_index=False).mean().plot('epoch', 'kappa')

    return


if __name__ == '__main__':
    
    Main()
    
    main()
