import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
from pre_process import resample_spacing, centering, rescale_intensity
import argparse
import matplotlib.pyplot as plt
import re

def resample_2d_im(image):
    try:
        image = sitk.Extract(image, (image.GetWidth(), image.GetHeight(), 0), (0, 0, 0))
    except Exception as e: print(e)
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    spacing = image.GetSpacing()
    print(spacing)
    resample.SetOutputSpacing([1.2, 1.2])
    resample.SetSize([256, 256])

    return resample.Execute(image)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def load_data_science_lv_bowl(dir_n, num, out_dir):
    """
    Args:
        dir_n: directory name of the folder containing images from all patients
        num: number of patients to process
    """
    import random
    fns = glob.glob(os.path.join(dir_n, '*'))
    random.shuffle(fns)
    fns = fns[:num]
    
    # plot all images
    row = int(np.sqrt(len(fns)))
    while len(fns) % row != 0:
        row-= 1
    col = int(len(fns)/row)
    fig, axes = plt.subplots(figsize=(col, row), nrows=row, ncols=col)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        try:
            im_list = list()
            sax = natural_sort(glob.glob(os.path.join(fns[i], 'study', 'sax*')))
            # get only the first time frame
            ref = None
            for s in sax:
                im_fn = glob.glob(os.path.join(s, '*.dcm'))[0]
                im = sitk.ReadImage(im_fn)
                if ref is None:
                    ref = im
                im = centering(im, ref)
                im = sitk.Extract(im, (im.GetWidth(), im.GetHeight(), 0), (0, 0, 0))
                im_list.append(im)
            volume = sitk.Cast(sitk.JoinSeries(im_list), sitk.sitkFloat32)
            out_dir_n = os.path.join(out_dir, os.path.basename(fns[i]))
            try:
                os.makedirs(out_dir_n)
            except Exception as e: print(e)
            volume_t = sitk.JoinSeries([volume, volume])
            sitk.WriteImage(volume_t, os.path.join(out_dir_n, os.path.basename(fns[i])+'_sa.nii.gz'))
            # resample to same in-plane resolution
            volume = resample_spacing(volume)[0]
            py_im = sitk.GetArrayFromImage(volume).transpose(2,1,0)
            py_im = rescale_intensity(py_im)
            py_im_slice = py_im[:,:,2]
            py_im_slice = (py_im_slice - np.mean(py_im_slice))/np.std(py_im_slice)
            #plot 
            ax.imshow(np.squeeze(py_im_slice), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e: print(e)

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0,wspace=0., hspace=0.)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Path to the input image data')
    parser.add_argument('--dataset', default='dslvbowl', help='Name of the dataset')
    parser.add_argument('--out_folder', help='Output folder name')
    parser.add_argument('--num', type=int, help='Number of patients to align.')
    
    args = parser.parse_args()

    if args.dataset == 'dslvbowl':
        load_data_science_lv_bowl(args.folder, args.num,args.out_folder)
    

