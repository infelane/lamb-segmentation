import os

import pandas as pd


def main():
    folder = 'C:/Users/admin/OneDrive - ugentbe/data/dataframes'

    l = []

    for file in os.listdir(folder):
        filename, ext = os.path.splitext(file)
        if not ext == '.csv':
            continue

        filepath = os.path.join(folder, file)

        df = pd.read_csv(filepath, delimiter=';')

        idx = df['kappa'].idxmax()

        kappa = df.loc[idx]['kappa']

        print(file)
        print(kappa)

        file_split = filename.split('_')

        if file_split[0] == 'ti':
            model_name = '-'.join(file_split[:2])
            rest = file_split[2:]
        else:
            model_name = file_split[0]
            rest = file_split[1:]

        data_i = {
            'fullname': file,
            'model':model_name,
            'rest':rest,
            'kappa_max': kappa
        }

        for r in rest:
            if r[:4] == 'data':
                data_i['data'] = r[4:]
            elif r[:3] == 'top':
                data_i['data'] += '-' + r
            elif r[:1] == 'd':
                data_i['d'] = int(r[1:])
            elif r[:1] == 'k':
                data_i['k'] = int(r[1:])
            elif r[:1] == 'n':
                data_i['n_per_class'] = int(r[1:])

        l.append(data_i)

    df_all = pd.DataFrame(l)

    def foo(df,
            n_per_class=80,
            d=None):

        df_filter = df
        df_filter = df_filter[(df_filter['data'] == 'val')]

        if n_per_class is not None:
            df_filter = df_filter[(df_filter['n_per_class'] == n_per_class)]

        if d is not None:
            df_filter = df_filter[(df_filter['d'] == d)]

        idx = df_filter['kappa_max'].idxmax()

        # print(idx)
        print(df.loc[idx])


    def foofoo(df):
        n_per_class = 80
        d = None
        foo(df, n_per_class=n_per_class, d=d)

    foofoo(df_all[(df_all['model'] == 'ti-unet')])
    foofoo(df_all[(df_all['model'] == 'simple')])
    foofoo(df_all[(df_all['model'] == 'unet')])

    # idx = df_all[(df_all['model'] == 'ti-unet') &
    #        (df_all['data'] == 'val')].idxmax()
    #
    # df_all[(df_all['model'] == 'ti-unet') &
    #        (df_all['data'] == 'val')].max()
    #
    # df_all[df_all['model']=='simple'].max()

    return


if __name__ == '__main__':
    main()
