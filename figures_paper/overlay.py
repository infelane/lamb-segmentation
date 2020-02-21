import numpy as np

from data.postprocessing import img_to_uint8


def semi_transparant(img_clean, y_bool, transparency = .5):
    """
    
    :param img_clean:
    :param y_bool:
    :param transparency: How transparent the overlay is. Between 0 (full) and 1 (invisible)
    :return:
    """
    
    assert len(img_clean.shape) == 3        # 3D
    assert img_clean.shape[-1] == 3    # RGB
    
    assert len(y_bool.shape) == 2           # 2D
    assert img_clean.shape[:2] == y_bool.shape
    assert y_bool.dtype == bool
    
    assert 0 <= transparency < 100
    
    img_overlay = img_to_uint8(img_clean)
    
    cyan = [0, 255, 255]

    img_overlay[y_bool, :] = transparency*img_overlay[y_bool, :] + (1-transparency)*np.asarray(cyan)
    
    return img_overlay
