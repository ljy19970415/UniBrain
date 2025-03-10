import cv2
import itertools
import numpy as np
import nibabel as nib
#from matplotlib import pyplot as plt
import os
import json
join = os.path.join

# COLOR_LIST = json.load(open('COLOR_LIST.json','r'))

def combine_data_seg_ana(data_volume, seg_volumes):
    # assign pre-defined color
    data_volume = data_norm(data_volume)
    data_volume = toRGB(data_volume)
    
    rate_for_image = 0.6

    data_seg_volumes = []
    
    for seg_volume in seg_volumes:
        seg_volume = toRGB(seg_volume)
        seg_volume = assign_color_my(seg_volume)
        data_seg_volume = np.zeros(shape=data_volume.shape, dtype=np.uint8)
        for i in range(data_volume.shape[0]):
            data_seg_volume[i,...] = cv2.addWeighted(data_volume[i,...], rate_for_image,
                                                    seg_volume[i,...], 1 - rate_for_image, 0)
        data_seg_volumes.append(data_seg_volume)
        
    return data_volume,data_seg_volumes

#%%
def data_norm(volume):
    # normalize the intensity into [0, 255]
    volume = (volume - np.min(volume)) * 255 / (np.max(volume) - np.min(volume))
    return volume

def toRGB(volume):
    # transform the volume from 1 channel to 3 channels
    volume = volume.astype(np.uint8)
    volume = np.stack((volume, )*3, axis=-1)
    ##############################################################
    # Note: Rotation here is only for the MSD. Revise it when it is needed.
    # volume shape: [X, Y, Channels, Z]
    # volume = np.transpose(volume, (1, 0, 3, 2))[::-1, ::-1, ...]
    ##############################################################
    return volume

def seg_norm(seg_volume):
    # assign the RGB channels and intensity for each seg of organs
    channels_candidate_list = []
    for i in range(1, 4):
        channels_candidate_list += itertools.combinations('012', i)
    print(channels_candidate_list)
    base_color = 128
    color_stride = (255 - base_color) * 3 // np.max(seg_volume)
    for i in range(1, np.max(seg_volume) + 1):
        color = color_stride * (i - 1)
        intensity = color % (255 - base_color) + base_color
        channels = channels_candidate_list[(i - 1) % len(channels_candidate_list)]
        channels = [int(x) for x in channels]
        print(i, intensity, channels)
        for j in range(3):
            if j in channels:
                seg_volume[:, :, j, :][seg_volume[:, :, j, :] == i] = intensity
            else:
                seg_volume[:, :, j, :][seg_volume[:, :, j, :] == i] = 0

    return seg_volume

def assign_color(seg_volume):
    # assign the RGB for each organ based on predefined color (COLOR_LIST)
    for i in range(1, np.max(seg_volume) + 1):
        for j in range(3):
            seg_volume[:, :, j, :][seg_volume[:, :, j, :] == i] = COLOR_LIST[i-1][j]

    return seg_volume

def combine_data_seg(data_volume, seg_volume):
    data_volume = data_norm(data_volume)
    data_volume = toRGB(data_volume)
    seg_volume = toRGB(seg_volume)
    seg_volume = seg_norm(seg_volume)
    data_seg_volume = np.zeros(shape=data_volume.shape, dtype=np.uint8)
    for i in range(data_volume.shape[-1]):
        rate_for_image = 0.7
        data_seg_volume[..., i] = cv2.addWeighted(data_volume[..., i], rate_for_image,
                                                  seg_volume[..., i], 1 - rate_for_image, 0)

    return data_seg_volume

def assign_color_my(seg_volume):
    # assign the RGB for each organ based on predefined color (COLOR_LIST)
    for i in range(1, np.max(seg_volume) + 1):
        for j in range(3):
            seg_volume[:, :, :, j][seg_volume[:, :, :, j] == i] = COLOR_LIST[i-1][j]

    return seg_volume

def combine_data_seg_my(data_volume):
    # assign pre-defined color

    data_volume = data_norm(data_volume)
    data_volume = toRGB(data_volume)

    return data_volume

# def combine_data_seg_my(data_volume, seg_volume, seg_volume2=None):
#     # assign pre-defined color
#     data_volume = data_norm(data_volume)
#     data_volume = toRGB(data_volume)
#     seg_volume = toRGB(seg_volume)
#     seg_volume = assign_color_my(seg_volume)
#     data_seg_volume = np.zeros(shape=data_volume.shape, dtype=np.uint8)

#     if seg_volume2 is not None:
#         seg_volume2 = toRGB(seg_volume2)
#         seg_volume2 = assign_color_my(seg_volume2)
#         data_seg_volume2 = np.zeros(shape=data_volume.shape, dtype=np.uint8)

#     rate_for_image = 0.6

#     for i in range(data_volume.shape[0]):
#         data_seg_volume[i,...] = cv2.addWeighted(data_volume[i,...], rate_for_image,
#                                                   seg_volume[i,...], 1 - rate_for_image, 0)
#         if seg_volume2 is not None:
#             data_seg_volume2[i,...] = cv2.addWeighted(data_volume[i,...], rate_for_image,
#                                                   seg_volume2[i,...], 1 - rate_for_image, 0)
    
#     if seg_volume2 is not None:
#         return data_volume,data_seg_volume, data_seg_volume2
#     else:
#         return data_volume,data_seg_volume

def imgs2video(imgs, zmin, show_name, save_name, fps=6):
    # define font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # fps = 2 frames per second
    size = (imgs.shape[0], imgs.shape[1])

    video_writer = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    # print('debug', save_name, size, video_writer.isOpened())
    for i in range(imgs.shape[-1]):
        # put text to image
        img_temp = np.fliplr(np.flipud(imgs[..., i]))
        frame = img_temp.copy()
        # img, text, coord, font, size, color, wide +1 aims to match the index in ITK-SNAP
        cv2.putText(frame, "%s: %d" % (show_name, zmin+i+1), (40, 40), font, 0.5, (255, 255, 255), 2)
        # video_writer.write(imgs[..., i])
        video_writer.write(frame)

    video_writer.release()

def find_seg_region(seg_volume, shift=3):
    # identify the z_min and z_max of a segmentation
    # seg.shape = (x,y,z); z should be the number of axial slices
    z_index = np.where(seg_volume==14)[2]
    z_min = np.max([np.min(z_index)-shift,0])
    z_max = np.min([np.max(z_index)+shift, seg_volume.shape[2]])
    
    return z_min, z_max
    
# img_path = r'./FLARE23Test400'
# seg_path = r'./0-GT'
# save_path = r'./video'
# names = os.listdir(seg_path)
# names.sort()

#from skimage import transform
#for i, name in enumerate(names):

def MR_nii2video(name):
    try: 
        data_volume = nib.load(join(img_path, name.split('.nii.gz')[0]+'_0000.nii.gz')).get_fdata()
        seg_volume = np.uint8(nib.load(join(seg_path, name)).get_fdata())
        z_min, z_max = find_seg_region(seg_volume)

        data_volume[data_volume>240] = 240
        data_volume[data_volume<-160] = -160
        
        # combine data and seg to one volume
        data_seg_volume = combine_data_seg_v2(data_volume[:,:, z_min:z_max], seg_volume[:,:, z_min:z_max])
        imgs2video(data_seg_volume, z_min, show_name=name.split(".nii.gz")[0], save_name=join(save_path, name.split(".nii.gz")[0] +".mp4"))
    except Exception as e:
        print(name, e, ' error!!!')


from multiprocessing import Pool
import time
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    time_start = time.time()
    with Pool(4) as p:
        _ = p.map(MR_nii2video, names)
    print(time.time() - time_start)
