import os
import numpy as np

from neuralNetwork.import_keras import ImageDataGeneratorOrig, NumpyArrayIterator, NumpyArrayIteratorPre, K, \
    array_to_img
from data.conversion_tools import y2bool_annot
from data.image_tools import ImgExt

# TODO this seems ugly import
from keras_preprocessing.image import np as np_random


def get_flow(x, y,
             batch_size=32,
             w_patch=10
             ):
    
    datagen = SegmentationDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        diagonal_flip=True,
    )

    flow = datagen.flow(x, y, y2bool_annot(y), w_in=w_patch,
                        w_out=w_patch, batch_size=batch_size)

    
    return flow


# Extension of original keras imagedatagenerator
class ImageDataGenerator(ImageDataGeneratorOrig):
    """
        diagonal_flip: whether to randomly flip images diagonally (use only for square images).
    """
    
    def __init__(self,
                 diagonal_flip=False,  # new addition (to include all 8 basic augmentations
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.diagonal_flip = diagonal_flip
    
    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        
        x = super().random_transform(x, seed=seed)
        
        if self.diagonal_flip:
            if np.random.random() < 0.5:
                x = _flip_diagonal(x, img_row_axis, img_col_axis)
        
        return x


class NumpyArrayCropIterator(NumpyArrayIterator):
    """Iterator yielding cropped data from a Numpy array.

    Arguments
        x: list of Numpy arrays of input data.
        y: list of Numpy arrays of targets data.
        image_data_generator: Instance of `SegmentationDataGenerator`
            to use for random transformations and normalization.
        mask: a binary 3D performance_map with all the pixels that can be cropped around
        w_in: integer, width of input patch
        w_out: integer, width of output patch
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in SegmentationDataGenerator.
    """
    def __init__(self, x, y,
                 image_data_generator,
                 mask,
                 w_in,
                 w_out,
                 batch_size=32, shuffle=False, seed=None, data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png', subset=None):
        
        # If single image, convert it to a list of length one
        if isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray)
            assert isinstance(mask, np.ndarray)
            x = [x]
            y = [y]
            mask = [mask]
        
        assert isinstance(x, (list, tuple))
        assert isinstance(y, (list, tuple))
        
        if y is not None:
            len_x, len_y, len_mask = len(x), len(y), len(mask)
            
            if len_x != len_y:
                raise ValueError('`x` (images tensor) and `y` (labels) '
                                 'should have the same length. '
                                 'Found: len(x) = %s, len(y) = %s' %
                                 (len_x, len_y))
            if len_x != len_mask:
                raise ValueError('`x` (images tensor) and `mask` (mask) '
                                 'should have the same length. '
                                 'Found: len(x) = %s, len(y) = %s' %
                                 (len_x, len_mask))
            
            for i in range(len_x):
                # assert x[i]
                assert isinstance(x[i], np.ndarray)
                assert isinstance(y[i], np.ndarray)
                assert isinstance(mask[i], np.ndarray)
                
                assert np.asarray(x[i]).shape[:-1] == np.asarray(y[i]).shape[:-1], (
                            '`x` (images tensor list) and `y` (labels list) '
                            'should have the same shape. '
                            'Found: x[i].shape = %s, y[i].shape = %s' %
                            (np.asarray(x[i]).shape[:-1], np.asarray(y[i]).shape[:-1]))
        
        if data_format is None:
            data_format = K.image_data_format()
        
        self.x = [np.asarray(x_i, dtype=K.floatx()) for x_i in x]
        
        for i in range(len(self.x)):
            x_i = self.x[i]
            if x_i.ndim != 3:
                raise ValueError('Input data in `NumpyArrayIterator` '
                                 'should be list of arrays with rank 3. You passed an array '
                                 'with shape', i, ':', x_i.shape)
        
        if y is not None:
            self.y = [np.asarray(y_i, dtype=K.floatx()) for y_i in y]
            
            for i in range(len(self.y)):
                y_i = self.y[i]
                
                assert y_i.shape[:2] == mask[i].shape, ('`y` (labels image) and `mask` (mask labels) '
                                                        'should have the same shape (except last channel). '
                                                        'Found: y.shape = %s, mask.shape = %s' %
                                                        (y_i.shape, np.asarray(mask[i]).shape))
                
                assert x_i.ndim == 3, ('Output data in `NumpyArrayIterator` '
                                       'should have rank 4. You passed an array '
                                       'with shape', i, ':', y_i.shape)
        
        else:
            self.y = None
        
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        
        if self.y is not None:
            self.xy_list = [np.concatenate([x_i, y_i], axis=-1) for x_i, y_i in zip(self.x, self.y)]
        else:
            self.xy_list = self.x
        
        self.xy_ext = [ImgExt(xy_i, ext=w_in // 2) for xy_i in self.xy_list]
        
        self.mask = mask
        self.w_in = w_in
        self.w_out = w_out
        
        assert self.w_in >= self.w_out, 'w_in should be >= w_out: {} vs {}'.format(self.w_in, self.w_out)
        
        if subset is not None:
            raise NotImplementedError(
                'Given the current implementation it is not possible to automatically split training and validation.'
                'This is because there is no standard way to split segmentation images with low amount of samples.')
        
        if self.data_format != 'channels_last':
            raise NotImplementedError('channels last only!', self.data_format)
        
        n = sum([np.count_nonzero(mask_i) for mask_i in mask])
        
        super(NumpyArrayIteratorPre, self).__init__(n, batch_size, shuffle, seed)
    
    # Overwrite
    def _set_index_array(self):
        
        index_array_lst = []
        for i in range(len(self.mask)):
            mask_i = self.mask[i]
            
            a = np.transpose(np.nonzero(mask_i))
            b = np.zeros((a.shape[0], 1), dtype=a.dtype)
            b[:] = i
            c = np.concatenate([b, a], axis=1)
            index_array_lst.append(c)
        
        self.index_array = np.concatenate(index_array_lst, 0)
        
        if self.shuffle:
            np_random.random.shuffle(self.index_array)  # only first axis is shuffled
    
    # updated
    def _get_batches_of_transformed_samples(self, index_array):
        
        w_in, w_out = self.w_in, self.w_out
        
        batch_x = np.zeros(tuple([len(index_array), w_in, w_in, self.x[0].shape[-1]]),
                           dtype=K.floatx())
        
        if self.y is not None:
            batch_y = np.zeros(tuple([len(index_array), w_out, w_out, self.y[0].shape[-1]]),
                               dtype=K.floatx())
        else:
            # TODO if y not given??? Might not need it
            
            raise NotImplementedError('y should be non-zero')
        
        if self.y is not None:
            f_x, f_y = self.x[0].shape[-1], self.y[0].shape[-1]
        else:
            f_x, f_y = self.x[0].shape[-1], 0
        
        xy_list = self.xy_list
        xy_ext_list = self.xy_ext
        
        for i, i_co in enumerate(index_array):
            i_image, i_h, i_w = i_co
            
            # precrop
            ext0 = (w_in) // 2
            
            # h0 = i_h - ext0
            # h1 = i_h + ext0 + 1
            # w0 = i_w - ext0
            # w1 = i_w + ext0 + 1
            # xy = xy_list[i_image][h0:h1, w0:w1, :]
            
            xy = xy_ext_list[i_image].get_crop(i_h, i_w, 1)
            
            xy = self.image_data_generator.random_transform(xy.astype(K.floatx()))
            
            w_precrop = ext0 * 2 + 1  # uneven
            h0_x = (w_precrop - w_in) // 2
            h1_x = w_in + h0_x
            x = xy[h0_x:h1_x, h0_x:h1_x, :f_x]
            # only standardize on x
            x = self.image_data_generator.standardize(x)
            
            # # TODO, solve out of bounds issues
            # # TODO now with workaround...
            # shape_x = x.shape
            # batch_x[i, :shape_x[0], :shape_x[1], :] = x
            batch_x[i] = x
            
            # TODO does this take long to calculate???
            if self.y is not None:
                h0_y = (w_precrop - self.w_out) // 2
                h1_y = self.w_out + h0_y
                
                y = xy[h0_y:h1_y, h0_y:h1_y, f_x:]
                
                # # TODO, solve out of bounds issues
                # # TODO now with workaround...
                # shape_y = y.shape
                #
                # batch_y[i, :shape_y[0], :shape_y[1], :] = y
                batch_y[i] = y
            
            # ext0 = self.w_in // 2
            # ext1 = self.w_in - ext0
            # h0 = i_h - ext0
            # h1 = i_h + ext1
            # w0 = i_w - ext0
            # w1 = i_w + ext1
            #
            # x = self.x[i_image][h0:h1, w0:w1, :]
            # x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            # x = self.image_data_generator.standardize(x)
        
        # if self.y is not None:
        #     for i, i_co in enumerate(index_array):
        #         i_image, i_h, i_w = i_co
        #         ext0 = self.w_out // 2
        #         ext1 = self.w_out - ext0
        #         h0 = i_h - ext0
        #         h1 = i_h + ext1
        #         w0 = i_w - ext0
        #         w1 = i_w + ext1
        #         # TODO y should do the same transform if it is an image :s
        #         # TODO idea: concatenate y to x, pass through transform and then split again and perhaps crop further.
        #         y = self.y[i_image][h0:h1, w0:w1, :]
        #         batch_y[i] = y
        
        if self.save_to_dir:
            for i, i_co in enumerate(index_array):
                i_image, i_h, i_w = i_co
                
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index1}_{index2}_{index3}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                                     index1=i_image,
                                                                                     index2=i_h,
                                                                                     index3=i_w,
                                                                                     hash=np.random.randint(1e4),
                                                                                     format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        
        if self.y is None:
            return batch_x
        
        return batch_x, batch_y


class SegmentationDataGenerator(ImageDataGenerator):
    def __init__(self,
                 validation_split=0.0,  # new addition (to include all 8 basic augmentations
                 **kwargs
                 ):
        
        if validation_split:
            raise NotImplementedError(
                'Given the current implementation it is not possible to automatically split training and validation.'
                'This is because there is no standard way to split segmentation images with low amount of samples.')
        
        super().__init__(**kwargs)
    
    def flow(self, x, y, mask, w_in, w_out, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):
        """Takes numpy data & label arrays, and generates batches of
            cropped/augmented/normalized data.

        Arguments
               x: data. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
               y: labels. Should have rank 4 and resolution as x.
               mask: a binary 3D performance_map with all the pixels that can be cropped around
               w_in: integer, width of input patch
               w_out: integer, width of output patch
               batch_size: int (default: 32).
               shuffle: boolean (default: True).
               seed: int (default: None).
               save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
               save_prefix: str (default: `''`). Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
               subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.

        Returns
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array of image data and
             `y` is a numpy array of corresponding labels.
        """
        
        if subset is not None:
            raise NotImplementedError(
                'Given the current implementation it is not possible to automatically split training and validation.'
                'This is because there is no standard way to split segmentation images with low amount of samples.')
        
        return NumpyArrayCropIterator(
            x, y, self, mask,
            w_in, w_out,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)
    
    def fit(self):
        raise NotImplementedError('Does not work yet with the random cropping!')


def _flip_diagonal(x, img_row_axis, img_col_axis):
    # Transpose along(img_col_axis, img_row_axis)
    
    axes = np.arange(len(np.shape(x)))
    axes[img_row_axis] = img_col_axis
    axes[img_col_axis] = img_row_axis
    axes = tuple(axes)
    x = np.asarray(x).transpose(axes)
    
    return x
