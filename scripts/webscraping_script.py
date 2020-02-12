import os

from PIL import Image

from data.webscraping import get_im


if __name__ == '__main__':
    modality = 'ir'
    modality = 'rgb'    # rgb before cleaning
    modality = 'xray'
    
    if modality == 'ir':
        if 0:
            panel_nr = 13
            resolution_scale = 17  # 17
        
        panel_nr = 19
        resolution_scale = 16
    elif modality == 'rgb':
        panel_nr = 19
        resolution_scale = 14
        
    elif modality == 'xray':
        panel_nr = 19
        resolution_scale = 15
    
    im = get_im(panel_nr=panel_nr, resolution_scale=resolution_scale, modality=modality, b_grid=True)

    if 0:
        folder = f'C:/Users/Laurens_laptop_w/OneDrive - UGent/data/{panel_nr}_small/webscrape'
    folder = f'/scratch/lameeus/data/ghent_altar/webscrape/{panel_nr}'
    assert os.path.exists(folder)
    Image.fromarray(im).save(os.path.join(folder, f'{modality}_{resolution_scale}.png'))

    print("Done")
