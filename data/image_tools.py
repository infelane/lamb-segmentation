import numpy as np


class ImgExt(object):
    """ the image_ext will now be slightly bigger"""
    
    def __init__(self, img, ext, edge='mirror', w=0):
        """
        :param img:
        :param ext:
        :param edge: mirror: mirror the edges, grey: edges set to 0.5
        """
        
        # super = self
        
        # a = ImgExt.__init__(self, img, ext, edge)
        # a = self
        
        assert isinstance(ext, (tuple, int))
        if type(ext) is int:  # convert to tuple
            ext = (ext, ext)
        
        # TODO remove int checking
        
        shape = np.shape(img)
        if type(ext) is tuple:
            assert len(ext) == 2
            assert ext[0] >= 0
            assert ext[1] >= 0
            
            shape_ext = [shape[0] + ext[0] + ext[1] + w, shape[1] + ext[0] + ext[1] + w, shape[2]]
        
        elif type(ext) is int:
            assert ext >= 0
            shape_ext = [shape[0] + 2 * ext + w, shape[1] + 2 * ext + w, shape[2]]
        
        else:
            raise TypeError
        
        if edge == 'grey':
            self._img_ext = np.ones(shape_ext) * 0.5
        else:
            self._img_ext = np.zeros(shape_ext)
        self.ext = ext
        
        if type(ext) is tuple:
            self._img_ext[ext[0]:ext[0] + shape[0], ext[0]:ext[0] + shape[1], :] = img
            if edge == 'mirror':
                # don't check if ext == 0
                # TODO check if everythin below is correct
                self._img_ext[:ext[0], ext[0]:shape[1] + ext[0], :] = np.flip(img[1:ext[0] + 1, :, :], axis=0)
                self._img_ext[shape[0] + ext[0]:, ext[0]:shape[1] + ext[0], :] = np.flip(
                    img[shape[0] - ext[1] - w - 1:shape[0] - 1, :, :], axis=0)
                
                self._img_ext[:, :ext[0], :] = np.flip(self._img_ext[:, ext[0] + 1: 2 * ext[0] + 1, :], axis=1)
                self._img_ext[:, shape[1] + ext[0]:, :] = np.flip(
                    self._img_ext[:, shape[1] + ext[0] - ext[1] - w - 1:shape[1] + ext[0] - 1, :],
                    axis=1)
        
        elif type(ext) is int:
            if ext > 0:
                self._img_ext[ext: shape[0] + ext, ext: + shape[1] + ext, :] = img
                if edge == 'mirror':
                    # self.__img_ext[:ext, ext:-ext, :] = img[ext - 1::-1, :, :]
                    # self.__img_ext[-ext:, ext:-ext, :] = img[:-ext - 1:-1, :, :]
                    #
                    # self.__img_ext[:, :ext, :] = self.__img_ext[:, 2 * ext - 1:ext - 1:-1, :]
                    # self.__img_ext[:, -ext:, :] = self.__img_ext[:, -ext - 1:-2 * ext - 1:-1, :]
                    
                    self._img_ext[:ext, ext:shape[1] + ext, :] = np.flip(img[1:ext + 1, :, :], axis=0)
                    self._img_ext[shape[0] + ext:, ext:shape[1] + ext, :] = np.flip(
                        img[shape[0] - ext - w - 1:shape[0] - 1, :, :], axis=0)
                    
                    self._img_ext[:, :ext, :] = np.flip(self._img_ext[:, ext + 1: 2 * ext + 1, :], axis=1)
                    self._img_ext[:, shape[1] + ext:, :] = np.flip(
                        self._img_ext[:, shape[1] - w - 1:shape[1] + ext - 1, :],
                        axis=1)
            
            else:
                self._img_ext[...] = img
    
    def get_crop(self, i_h, i_w, w):
        ext = self.ext
        if type(ext) is tuple:
            return self()[i_h: i_h + w + ext[0] + ext[1], i_w: i_w + w + ext[0] + ext[1], :]
        elif type(ext) is int:
            return self()[i_h: i_h + w + 2 * ext, i_w: i_w + w + 2 * ext, :]
        
        # TODO replace v1. This one can handle a bigger range of getting
    
    # def get_crop2(self, i_h, i_w, w):
    #     ext = self.ext
    #     if type(ext) is tuple:
    #         return self()[i_h: i_h + w + ext[0] + ext[1], i_w: i_w + w + ext[0] + ext[1], :]
    #     elif type(ext) is int:
    #         return self()[i_h: i_h + w + 2 * ext, i_w: i_w + w + 2 * ext, :]
    
    def get_extended(self):
        return self._img_ext
    
    def __call__(self, *args, **kwargs):
        return self._img_ext

