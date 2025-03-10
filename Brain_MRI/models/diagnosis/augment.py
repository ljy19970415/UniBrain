import skimage.transform as transform
import scipy
import numpy as np
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import cv2
from collections import OrderedDict

def nnUNet_resample_and_normalize(data, new_shape, normalize=True,do_separate_z=True,resize_mode='interp'):
    data = nnUNet_resample(data, new_shape,do_separate_z=do_separate_z,resize_mode=resize_mode)
    if normalize:
        mn = data.mean()
        std = data.std()
        data = (data - mn) / (std + 1e-8)
    return data

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def nnUNet_resize(data, new_shape, do_separate_z=True, is_seg=False, axis=2, order=3, order_z=0):
    assert len(data.shape) == 3, "data must be (x, y, z)"
    assert len(new_shape) == len(data.shape)

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
        order = 1
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            #print("separate z, order in z is", order_z, "order inplane is", order)
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []

            reshaped_data = []
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                else:
                    reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
            reshaped_data = np.stack(reshaped_data, axis)
            # print("reshaped_data",reshaped_data.shape)
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                reshaped_data = map_coordinates(reshaped_data, coord_map, order=order_z, mode='nearest').astype(dtype_data)
                
                #print("shape[axis] != new_shape[axis]",reshaped_data.shape)
                # else:
                #     unique_labels = np.unique(reshaped_data)
                #     reshaped = np.zeros(new_shape, dtype=dtype_data)

                #     for i, cl in enumerate(unique_labels):
                #         reshaped_multihot = np.round(
                #             map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                #                             mode='nearest'))
                #         reshaped[reshaped_multihot > 0.5] = cl
                #     reshaped_final_data.append(reshaped[None].astype(dtype_data))
            # else:
            #     reshaped_data = reshaped_data[None].astype(dtype_data)
        else:
            reshaped_data = resize_fn(data, new_shape, order, **kwargs).astype(dtype_data)
            # print("no separate z, order", reshaped_data.shape)
        return reshaped_data.astype(dtype_data)
    else:
        print("no resampling necessary",data.shape)
        return data

def nnUNet_resample(data, new_shape, do_separate_z=True, resize_mode="normal", axis=2, order=3, order_z=0):
    assert len(data.shape) == 3, "data must be (x, y, z)"
    assert len(new_shape) == len(data.shape)

    if resize_mode=='interp':
        resize_fn = cv2.resize
        kwargs = {'interpolation':cv2.INTER_CUBIC}
        order_z = 1
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            #print("separate z, order in z is", order_z, "order inplane is", order)
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []

            reshaped_data = []
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                else:
                    reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
            reshaped_data = np.stack(reshaped_data, axis)
            # print("reshaped_data",reshaped_data.shape)
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                reshaped_data = map_coordinates(reshaped_data, coord_map, order=order_z,
                                                            mode='nearest').astype(dtype_data)
                #print("shape[axis] != new_shape[axis]",reshaped_data.shape)
                # else:
                #     unique_labels = np.unique(reshaped_data)
                #     reshaped = np.zeros(new_shape, dtype=dtype_data)

                #     for i, cl in enumerate(unique_labels):
                #         reshaped_multihot = np.round(
                #             map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                #                             mode='nearest'))
                #         reshaped[reshaped_multihot > 0.5] = cl
                #     reshaped_final_data.append(reshaped[None].astype(dtype_data))
            # else:
            #     reshaped_data = reshaped_data[None].astype(dtype_data)
        else:
            reshaped_data = resize_fn(data, new_shape, order, **kwargs).astype(dtype_data)
            # print("no separate z, order", reshaped_data.shape)
        return reshaped_data.astype(dtype_data)
    else:
        print("no resampling necessary",data.shape)
        return data

def downscale(image, shape):
    'For upscale, anti_aliasing should be false'
    return transform.resize(image, shape, mode='constant', anti_aliasing=True)


def random_rotations(img, min_angle=-90, max_angle=90):
    """
    Rotate 3D image randomly
    """
    assert img.ndim == 3, "Image must be 3D"
    rotation_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle+1)
    axes_random_id = np.random.randint(low=0, high=len(rotation_axes))
    axis = rotation_axes[axes_random_id] # Select a random rotation axis
    return scipy.ndimage.rotate(img, angle, axes=axis)


def random_zoom(img,min=0.7, max=1.2):
    """
    Generate random zoom of a 3D image
    """
    zoom = np.random.sample()*(max - min) + min # Generate random zoom between min and max
    zoom_matrix = np.array([[zoom, 0, 0, 0],
                            [0, zoom, 0, 0],
                            [0, 0, zoom, 0],
                            [0, 0, 0, 1]])
    
    return scipy.ndimage.interpolation.affine_transform(img, zoom_matrix)


def random_flip(img):
    """
    Flip image over a random axis
    """
    axes = [0, 1, 2]
    rand_axis = np.random.randint(len(axes))
    img = img.swapaxes(rand_axis, 0)
    img = img[::-1, ...]
    img = img.swapaxes(0, rand_axis)
    img = np.squeeze(img)
    return img


def random_shift(img, max=0.4):
    """
    Random shift over a random axis
    """
    (x, y, z) = img.shape
    (max_shift_x, max_shift_y, max_shift_z) = int(x*max/2),int(y*max/2), int(z*max/2)
    shift_x = np.random.randint(-max_shift_x, max_shift_x)
    shift_y = np.random.randint(-max_shift_y,max_shift_y)
    shift_z = np.random.randint(-max_shift_z,max_shift_z)

    translation_matrix = np.array([[1, 0, 0, shift_x],
                                   [0, 1, 0, shift_y],
                                   [0, 0, 1, shift_z],
                                   [0, 0, 0, 1]
                                   ])

    return scipy.ndimage.interpolation.affine_transform(img, translation_matrix)