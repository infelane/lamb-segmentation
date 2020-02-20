from data.image_tools import DataInference
from data.conversion_tools import img2batch, batch2img


def inference(model, img_in, w=None, w_ext=0):
    
    if w is None:
        # Try to get output w from model
        w = model.output_shape[-2]
        assert isinstance(w, int)
        
    data = DataInference(batch2img(img_in), w=w)
    
    # get x_in from img_in
    x_in = data.img_to_x(batch2img(img_in), ext=(w_ext//2, w_ext - w_ext//2))
    
    # model prediction
    y = model.predict(x_in)
    
    # post processing
    img_out = data.y_to_img(y)
    
    return img_out
