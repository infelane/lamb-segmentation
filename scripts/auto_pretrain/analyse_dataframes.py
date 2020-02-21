import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    
    folder = '/home/lameeus/data/ghent_altar/dataframes/'
    file = 'pretrained_unet_10lamb_kfold_single.csv'
    
    file2 = 'tiunet_10lamb_kfold_single.csv'
    
    df_unet_pretrained_fixed = pd.read_csv(os.path.join(folder, file), sep=';')
    df_tiunet_single = pd.read_csv(os.path.join(folder, file2), sep=',')
    
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

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    
    c = df_tiunet_single[df_tiunet_single['i_fold']==0]
    ax.plot_trisurf(c.k, c.epoch, c.kappa,
                    # cmap=cm.jet,
                    # linewidth=0.2
                    )
    plt.show()
    
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
    main()
