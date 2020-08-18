import os
import sys
import glob
#import numpy as np
import SimpleITK as sitk
from pre_process import centering
from pre_process import affine_usage
import argparse
#import matplotlib.pyplot as plt
import re
import math
import traceback



def resample_2d_im(image):
    try:
        image = sitk.Extract(image, (image.GetWidth(), image.GetHeight(), 0), (0, 0, 0))
    except Exception as e:
        print(e)
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    spacing = image.GetSpacing()
    print(spacing)
    resample.SetOutputSpacing([1.2, 1.2])
    resample.SetSize([256, 256])
    return resample.Execute(image)

def writeParameterMap(parameter_map, fn):
    for i, para_map in enumerate(parameter_map):
        para_map_fn = os.path.splitext(fn)[0]+'_%d.txt'%i
        sitk.WriteParameterFile(para_map, para_map_fn)

def readParameterMap(fn):
    fns = sorted(glob.glob(os.path.splitext(fn)[0]+"*"))
    if len(fns) == 0:
        raise IOError("No Transformation file found")
    map_list = list()
    for para_map_fn in fns:
        map_list.append(sitk.ReadParameterFile(para_map_fn))
    parameter_map = tuple(map_list)
    return parameter_map
p_list = []

def registration(im, im2, path_name):
    global p_list
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(im)
    elastixImageFilter.SetMovingImage(im2)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), path_name)
    params = elastixImageFilter.GetTransformParameterMap()
    registration_params = params[0]['TransformParameters']
    registration_params = list(registration_params)
    registration_params = registration_params[1:3]


    p_list = p_list + registration_params


    return elastixImageFilter.GetResultImage()
    # fixedImage3d = sitk.ReadImage(im)
    # fixedImage2d = sitk.Extract(fixedImage3d, (fixedImage3d.GetWidth(), fixedImage3d.GetHeight(), 0), (0,0,0))
    # fixedimage2dfloat = sitk.Cast(fixedImage2d, sitk.sitkFloat32)
    # movingImage3d = sitk.ReadImage(im2)
    # movingImage2d = sitk.Extract(movingImage3d, (movingImage3d.GetWidth(), movingImage3d.GetHeight(), 0), (0,0,0))
    # movingimage2dfloat = sitk.Cast(movingImage2d, sitk.sitkFloat32)
    # parameterMap = sitk.GetDefaultParameterMap("rigid")
    # parameterMap["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
    # parameterMap["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
    #
    # elastix = sitk.SimpleElastix()
    # elastix.FixedImage(fixedimage2dfloat)
    # elastix.MovingImage(movingimage2dfloat)
    # elastix.SetParameterMap(parameterMap)
    # elastix.Execute()
    # SimpleElastix.GetResultImage()


def g_spacing(image):
    image = sitk.ReadImage(image)
    #reader.SetFileName(im)
    # reader.ReadImageInformation()
    #print(im.GetMetaDataKeys())
    imspacing = image.GetMetaData('0018|0050')
    #print(imspacing)
    imspacing = [float(imspacing)]
    return imspacing

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_data_science_lv_bowl(dir_n, num, out_dir, b):
    global p_list
    reg_list = []
    xlist = []
    ylist = []
    if b == True:
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
        # row = int(math.sqrt(len(fns)))
        # while len(fns) % row != 0:
        #     row -= 1
        # col = int(len(fns) / row)
        #fig, axes = plt.subplots(figsize=(col, row), nrows=row, ncols=col)
        #axes = axes.ravel()

        for i, fn in enumerate(fns):
            volume_list = []
            #writeParameterMap(readParameterMap(fn), fn)
                #try:
            sax = natural_sort(glob.glob(os.path.join(fns[i], 'study', 'sax*')))
            # get only the first time frame
            im_num = natural_sort(glob.glob(os.path.join(sax[0], '*.dcm')))
            ref = None

            for x in range(len(im_num)):

                im_list = []
                for index, s in enumerate(sax):

                    if index == 0:

                        im_fn = natural_sort(glob.glob(os.path.join(s, '*.dcm')))
                        im_one = im_fn[x]
                        im_one = sitk.ReadImage(im_one)
                        if ref is None:
                            ref = im_one
                        im_one = centering(im_one, ref)
                        im_one = sitk.Extract(im_one, (im_one.GetWidth(), im_one.GetHeight(), 0), (0, 0, 0))
                        # im_one, p1 = affine_usage(im_one)
                        # reg_list = reg_list + p1
                    else:

                        im_fn = natural_sort(glob.glob(os.path.join(s, '*.dcm')))
                        im_two = im_fn[x]
                        spacing = g_spacing(im_two)

                        im_two = sitk.ReadImage(im_two)
                        if ref is None:
                            ref = im_two
                        im_two = centering(im_two, ref)
                        im_two = sitk.Extract(im_two, (im_two.GetWidth(), im_two.GetHeight(), 0), (0, 0, 0))
                        im_two, p2 = affine_usage(im_two)

                        path_name = os.path.join(out_dir, os.path.basename(fns[i]) + '_sa2.nii.gz')
                        im = registration(im_one, im_two, path_name)
                        im_one = im
                        im_list.append(im)
                        reg_list = reg_list + p2

                volume = sitk.Cast(sitk.JoinSeries(im_list), sitk.sitkFloat32)
                v_spacing = volume.GetSpacing()
                v_spacing = list(v_spacing[0:2])
                volume.SetSpacing(v_spacing + spacing)
                volume_list.append(volume)

            out_dir_n = os.path.join(out_dir, os.path.basename(fns[i]))
            try:
                os.makedirs(out_dir_n)
            except OSError as e:
                print(e)

            volume_t = sitk.JoinSeries(volume_list)
            sitk.WriteImage(volume_t, os.path.join(out_dir_n, os.path.basename(fns[i]) + '_sa.nii.gz'))

            c = []
            for i in range(len(p_list)):
                difference = float(p_list[i])-reg_list[i]
                c.append(difference)
            xtotal = 0.0
            ytotal = 0.0
            for x in range(len(c)):
                if (x%2) ==0:
                    xtotal += abs(c[x])
                else:
                    ytotal += abs(c[x])
            xavg = xtotal/len(c)
            yavg = ytotal/len(c)
            print("average: ", xavg, " ", yavg)
            ylist.append(yavg)
            xlist.append(xavg)
            # print("Result: ")
            # for x in range(len(c)):
            #     print(c[x])
            # resample to same in-plane resolution
            # volume = resample_spacing(volume)[0]
            # py_im = sitk.GetArrayFromImage(volume).transpose(2, 1, 0)
            # py_im = rescale_intensity(py_im)
            # py_im_slice = py_im[:, :, 2]
            # py_im_slice = (py_im_slice - np.mean(py_im_slice)) / np.std(py_im_slice)
            # # plot
            # ax.imshow(np.squeeze(py_im_slice), cmap='gray')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # except Exception as e:
            #     print(e)
                #traceback.print_exc()
        #plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0., hspace=0.)
        #plt.show()
    else:
        import random
        fns = glob.glob(os.path.join(dir_n, '*'))
        random.shuffle(fns)
        fns = fns[:num]
        # plot all images
        # row = int(math.sqrt(len(fns)))
        # while len(fns) % row != 0:
        #     row -= 1
        # col = int(len(fns) / row)
        #fig, axes = plt.subplots(figsize=(col, row), nrows=row, ncols=col)
        #axes = axes.ravel()

        for i, fn in enumerate(fns):
            volume_list = []
            try:
                sax = natural_sort(glob.glob(os.path.join(fns[i], 'study', 'sax*')))
                # get only the first time frame
                im_num = natural_sort(glob.glob(os.path.join(sax[0], '*.dcm')))
                ref = None
                for x in range(len(im_num)):
                    im_list = []
                    for s in sax[0:2]:
                        im_fn = natural_sort(glob.glob(os.path.join(s, '*.dcm')))
                        im_fn = im_fn[x]
                        spacing = g_spacing(im_fn)
                        im = sitk.ReadImage(im_fn)
                        if ref is None:
                            ref = im
                        im = centering(im, ref)
                        im = sitk.Extract(im, (im.GetWidth(), im.GetHeight(), 0), (0, 0, 0))
                        im, p2 = affine_usage(im)
                        print("p2")
                        print(p2)
                        reg_list = reg_list + p2
                        im_list.append(im)
                    volume = sitk.Cast(sitk.JoinSeries(im_list), sitk.sitkFloat32)
                    v_spacing = volume.GetSpacing()
                    v_spacing = list(v_spacing[0:2])
                    volume.SetSpacing(v_spacing + spacing)
                    volume_list.append(volume)
                out_dir_n = os.path.join(out_dir, os.path.basename(fns[i]))
                try:
                    os.makedirs(out_dir_n)
                except OSError as e:
                    print(e)
                volume_t = sitk.JoinSeries(volume_list)
                sitk.WriteImage(volume_t, os.path.join(out_dir_n, os.path.basename(fns[i]) + '_sa.nii.gz'))
                print("Center adjusted parameters: ")
                print(reg_list)
                print(type(reg_list))
                # resample to same in-plane resolution
                # volume = resample_spacing(volume)[0]
                # py_im = sitk.GetArrayFromImage(volume).transpose(2, 1, 0)
                # py_im = rescale_intensity(py_im)
                # py_im_slice = py_im[:, :, 2]
                # py_im_slice = (py_im_slice - np.mean(py_im_slice)) / np.std(py_im_slice)
                # # plot
                # ax.imshow(np.squeeze(py_im_slice), cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
            except Exception as e:
                print(e)
                traceback.print_exc()
    print(xlist)
    print(ylist)

        #plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0., hspace=0.)
        #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Path to the input image data')
    parser.add_argument('--dataset', default='dslvbowl', help='Name of the dataset')
    parser.add_argument('--out_folder', help='Output folder name')
    parser.add_argument('--num', type=int, help='Number of patients to align.')
    parser.add_argument('--registration', type=str2bool, nargs='?', const=True, default=False, help='True/False use registration')

    args = parser.parse_args()

    if args.dataset == 'dslvbowl':
        load_data_science_lv_bowl(args.folder, args.num, args.out_folder, args.registration)