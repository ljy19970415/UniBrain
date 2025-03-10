import cv2
from add_mask_img import *
import imageio
import numpy as np
import nibabel as nib
import json
import os
import SimpleITK as sitk

def nib_load(file_name,component=0):
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    if data.ndim>3:
        data=data[:,:,:,component]
    proxy.uncache()
    return data

def mask2jpg_from_path(img_path, out_path):

    # img_array = np.array(nib_load(img_path), dtype='float32', order='C')

    original_img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(original_img).astype('float32')
    print(img_array.shape)
    z = img_array.shape[0]
    img_array = combine_data_seg_my(img_array)
    print(img_array.shape)
    # img_dir = os.path.join(save_folder,"img")
    # mask_dir = os.path.join(save_folder,"mask")
    # os.makedirs(img_dir,exist_ok=True)
    # os.makedirs(mask_dir,exist_ok=True)
    slice = z//2
    # temp1 = np.transpose(img_array[:,:,slice],(1,0,2))
    temp1 = img_array[slice,::-1,:,:]
    cv2.imwrite(out_path,temp1)

# def nib_load(file_name,component=0):
#     proxy = nib.load(file_name)
#     data = proxy.get_fdata()
#     if data.ndim>3:
#         data=data[:,:,:,component]
#     proxy.uncache()
#     return data

# def nii2jpg(input_path,slice,savepath):
#     img_array = np.array(nib_load(input_path), dtype='float32', order='C')
#     # img_array = np.array(nib_load(input_path), dtype=np.uint8, order='C')
#     imageio.imwrite(savepath,img_array[:,:,slice])

# def DWI2nii(rootdir):
#     components = json.load(open("E:\博士\博士科研\MRI_Pretrain_Model\\6th记录\merge\Select.json",'r'))
#     for root1, dirs, _ in os.walk(rootdir):
#         for dir in dirs:
#             fid = dir
#             DWI_path = os.path.join(root1,fid,components[fid]['DWI'].split('/')[-1])
#             print(fid,DWI_path)
#             proxy = nib.load(DWI_path)
#             data = proxy.get_fdata()
#             if data.ndim>3:
#                 data=data[:,:,:,components[fid]['component']]
#                 nib.save(nib.Nifti1Image(data, proxy.affine), DWI_path)

if __name__=='__main__':
    input_dir = "/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/data/img/1"
    output_dir = "/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/data/icon/1_prev"
    for root,dirs,files in os.walk(input_dir):
        for f in files:
            print(f)
            uid = f.split('.')[0]
            input_path = os.path.join(root,f)
            output_path = os.path.join(output_dir,uid+'.jpg')
            mask2jpg_from_path(input_path, output_path)

    # inputpath = 'E:\\niidata\each_dis\meni\\a107226024'
    # modals = ['DWI','T1WI','T2WI','T2FLAIR']
    # slice = [13,15,15,15]
    # for idx,modal in enumerate(modals):
    #     nii2jpg(os.path.join(inputpath,modal+'.nii.gz'),slice[idx],os.path.join(inputpath,modal+'.jpg'))