import os

from PIL import Image

from data.webscraping import get_im


if __name__ == '__main__':
    modality = 'ir'
    panel_nr = 13
    resolution = 17
    im = get_im(panel_nr=panel_nr, resolution=resolution, modality=modality, b_grid=True)

    folder = f'C:/Users/Laurens_laptop_w/OneDrive - UGent/data/{panel_nr}_small/webscape'
    assert os.path.exists(folder)
    Image.fromarray(im).save(os.path.join(folder, f'{modality}_{resolution}.png'))

    print("Done")
