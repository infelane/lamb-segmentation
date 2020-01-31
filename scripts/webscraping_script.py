from data.webscraping import get_im


if __name__ == '__main__':
    im = get_im(panel_nr=13, resolution=17, modality='ir', b_grid=True)
