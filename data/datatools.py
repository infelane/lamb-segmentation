import os

import numpy as np
import pandas as pd
from PIL import Image


def imread(path):
    im = np.array(Image.open(path))
    return im
    
    
def imsave(path, array, b_check_duplicate=True):
    
    if b_check_duplicate:
        if os.path.exists(path):
            feedback = input(f"file exists ({path}).\nType 'Y' if you want to overwrite it").lower()
            if feedback != 'y':
                print(f'Not saved: {path}')
                return

    Image.fromarray(array).save(path)
    return


def pandas_save(path, df, append=False, overwrite=False, index = False, sep=';', *args, **kwargs):
    
    if not overwrite and os.path.exists(path):
        if append:
            columns = pd.read_csv(path, nrows=0, sep=sep).columns
            
            assert set(df.columns) == set(columns), f'{df.columns}\n{columns}'

            df.to_csv(path, mode='a', header=False,
                      index=index, columns=columns, sep=sep
                      )
 
        else:
            print(f'File already exists: {path}\nNot overwriting it')
            return -1
    
    else:
        columns = df.columns
        df.to_csv(path, index=index, columns=columns, sep=sep, *args, **kwargs)
