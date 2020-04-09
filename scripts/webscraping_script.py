import os

import numpy as np

from data.webscraping import get_im
from data.datatools import imsave


if __name__ == '__main__':
    # modality = 'ir'
    # modality = 'rgb'    # rgb before cleaning
    modality = 'xray'
    # modality = 'irr'

    panel_nr = 13
    
    resolution_scale = 15
    
    if modality == 'ir':
        if 0:
            panel_nr = 13
            resolution_scale = 17  # 17

        resolution_scale = 16
    elif modality == 'rgb':
        resolution_scale = 14
        
    elif modality == 'xray':
        resolution_scale = 15
        
    if (modality == 'xray') & (panel_nr == 13):
        resolution_scale = 16
    if (modality == 'irr') & (panel_nr == 13):
        resolution_scale = 15
        
    im = get_im(panel_nr=panel_nr, resolution_scale=resolution_scale, modality=modality, b_grid=True)

    if modality in ('irr', 'ir', 'xray'):
        if len(im.shape) == 3:  # In case the grayscale is loaded as 3 colour channels.
            im = np.mean(im, axis=2)

    if os.name == 'nt':  # Laptop windows
        if panel_nr == 13:
            folder_save = f'C:/Users/Laurens_laptop_w/OneDrive - UGent/data/13_small/webscrape'
        else:
            folder_save = None
            raise NotImplementedError()
    else:   # Desktop Linux
        if 0:
            folder_save = f'C:/Users/Laurens_laptop_w/OneDrive - UGent/data/{panel_nr}_small/webscrape'
        folder_save = f'/scratch/lameeus/data/ghent_altar/webscrape/{panel_nr}'

    assert os.path.exists(folder_save)
    path_save = os.path.join(folder_save, f'{modality}_{resolution_scale}.png')
    imsave(path_save, im, b_check_duplicate=True)

    print("Done")
