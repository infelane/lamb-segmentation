import numpy as np

from data.postprocessing import img_to_uint8


def semi_transparant(img_clean, y_bool, transparency=.5,
                     color1='cyan',
                     color2=None,
                     transparancy2=None):
    """
    
    :param img_clean:
    :param y_bool:
    :param transparency: How transparent the overlay is. Between 0 (full) and 1 (invisible)
    :param color2: If you want there to be a background color
    :return:
    """
    
    assert len(img_clean.shape) == 3        # 3D
    assert img_clean.shape[-1] == 3    # RGB
    
    assert len(y_bool.shape) == 2           # 2D
    assert img_clean.shape[:2] == y_bool.shape
    assert y_bool.dtype == bool
    
    assert 0 <= transparency < 100
    
    img_overlay = img_to_uint8(img_clean)

    color1 = color1.lower()
    if color1 == 'cyan':
        c = [0, 255, 255]
    elif color1 == 'green':
        c = [0, 255, 0]
    else:
        raise ValueError(f'Not implemented color: {color1}')

    img_overlay[y_bool, :] = transparency*img_overlay[y_bool, :] + (1-transparency)*np.asarray(c)

    if color2 is not None:
        color2 = color2.lower()
        if color2 in ['grey', 'gray']:
            c2 = [125, 125, 125]
        else:
            raise ValueError(f'Not implemented color: {color2}')

        if transparancy2 is not None:
            t = transparancy2
        else:
            t = transparency
        img_overlay[~y_bool, :] = t * img_overlay[~y_bool, :] + \
                                     (1 - t) * np.asarray(c2)
    
    return img_overlay
