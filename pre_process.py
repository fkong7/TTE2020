# import numpy as np
import SimpleITK as sitk
import random


def reference_image_build(spacing, size, direction, template_size, dim):
    # template size: image(array) dimension to resize to: a list of three elements
    reference_spacing = np.array(size) / np.array(template_size) * np.array(spacing)
    reference_spacing[0] = 1.2
    reference_spacing[1] = 1.2
    reference_image = sitk.Image(template_size, 0)
    reference_image.SetOrigin(np.zeros(3))
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(direction)
    return reference_image


def centering(img, ref_img, order=1):
    dimension = img.GetDimension()
    transform = sitk.AffineTransform(dimension)
    list1 = img.GetOrigin()
    list2 = ref_img.GetOrigin()
    list3 = [None] * 3
    for i in range(3):
        list3[i] = list1[i] - list2[i]
    transform.SetTranslation(list3)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    size = img.GetSize()
    size = [s / 2 for s in size]
    img_center = img.TransformContinuousIndexToPhysicalPoint(size)
    size_ref = ref_img.GetSize()
    size_ref = [s / 2 for s in size_ref]
    reference_center = ref_img.TransformContinuousIndexToPhysicalPoint(size_ref)
    list1 = transform.GetInverse().TransformPoint(img_center)
    list2 = reference_center
    list3 = [None] * 3
    for i in range(3):
        list3[i] = list1[i] - list2[i]
    centering_transform.SetOffset(list3)
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    return transform_func(img, ref_img, centered_transform, order)


def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
    dim = ref_img.GetDimension()
    affine = sitk.AffineTransform(dim)
    if target is None:
        target = np.eye(dim)

    ori = np.reshape(orig_direction, np.eye(dim).shape)
    target = np.reshape(target, np.eye(dim).shape)
    affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize()) / 2.0))
    return transform_func(image, ref_img, affine, order)


def transform_func(image, reference_image, transform, order=1):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if order == 1:
        interpolator = sitk.sitkLinear
    elif order == 2:
        interpolator = sitk.sitkBSpline
    elif order == 0:
        interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    try:
        resampled = sitk.Resample(image, reference_image, transform,
                                  interpolator, default_value)
    except Exception as e:
        print(e)

    return resampled


def resample_spacing(sitkIm, resolution=0.5, dim=3, template_size=(256, 256), order=1):
    if type(sitkIm) is str:
        image = sitk.ReadImage(sitkIm)
    else:
        image = sitkIm
    orig_direction = image.GetDirection()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size * (orig_spacing / np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    template_size = (template_size[0], template_size[1], int(orig_size[-1]))
    ref_img = reference_image_build(resolution, new_size, image.GetDirection(), template_size, dim)
    centered = centering(image, ref_img, order)
    transformed = isometric_transform(centered, ref_img, orig_direction, order)
    return transformed, ref_img


def resize_to_size(image, size=(256, 256), order=1):
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = [int(size[0]), int(size[1]), int(orig_size[-1])]
    new_spacing = orig_spacing * orig_size / np.array(new_size)
    if order == 1:
        interpolator = sitk.sitkLinear
    elif order == 2:
        interpolator = sitk.sitkBSpline
    elif order == 0:
        interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    fltr = sitk.ResampleImageFilter()
    fltr.SetSize(new_size)
    fltr.SetOutputSpacing(new_spacing)
    fltr.SetOutputOrigin(image.GetOrigin())
    fltr.SetOutputDirection(image.GetDirection())
    fltr.SetInterpolator(interpolator)
    image = fltr.Execute(image)
    return image


def resample_scale(sitkIm, ref_img, gt_img=None, scale_factor=1., order=1):
    sitkIm.SetDirection(np.eye(3).ravel())
    ref_img.SetDirection(np.eye(3).ravel())
    gt_img.SetDirection(np.eye(3).ravel())
    dim = sitkIm.GetDimension()
    affine = sitk.AffineTransform(dim)
    scale = np.array(ref_img.GetDirection())
    scale = np.reshape(scale, (dim, dim))
    scale[:, 0] *= 1. / scale_factor
    scale[:, 1] *= 1. / scale_factor

    if gt_img is not None:
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(gt_img, sitk.sitkInt32))
        center = stats.GetCentroid(1)
    else:
        center = sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize()) / 2.0)
    affine.SetMatrix(scale.ravel())
    affine.SetCenter(center)
    transformed = transform_func(sitkIm, ref_img, affine, order)
    return transformed


def swap_labels(labels):
    unique_label = np.unique(labels)
    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        label = unique_label[i]
        newl = new_label[i]
        labels[labels == label] = newl
    return labels


def swap_labels_back(labels, pred):
    unique_label = np.unique(labels)
    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        pred[pred == i] = unique_label[i]
    return pred


def rescale_intensity(slice_im):
    if type(slice_im) != np.ndarray:
        raise RuntimeError("Input image is not numpy array")
    # upper = np.percentile(slice_im, 90)
    upper = np.percentile(slice_im, 99)
    lower = np.percentile(slice_im, 20)
    slice_im[slice_im > upper] = upper
    slice_im[slice_im < lower] = lower
    slice_im -= lower
    rng = upper - lower
    slice_im = slice_im / rng * 2.
    slice_im -= 1.
    # slice_im = (slice_im - np.mean(slice_im))/np.std(slice_im)
    return slice_im


def swap_low_freq(im1, im2, beta):
    """
    Change the low frequency of im2 with that of im1
    Beta: ratio between the swaped region and the image dimension
    """
    # im1 = denoise(im1, 10, 0.125)
    # im2 = denoise(im2, 10, 0.125)
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1 - np.min(im1)
    im2 = im2 - np.min(im2)
    im1_fft = np.fft.fftshift(np.fft.fft2(im1))
    im2_fft = np.fft.fftshift(np.fft.fft2(im2))
    change = beta * np.array(im2_fft.shape)
    up0 = int(im2.shape[0] / 2 - change[0] / 2)
    down0 = int(im2.shape[0] / 2 + change[0] / 2)
    up1 = int(im2.shape[1] / 2 - change[1] / 2)
    down1 = int(im2.shape[1] / 2 + change[1] / 2)
    # im2_fft[up0:down0, up1:down1] = 0.
    im2_fft[up0:down0, up1:down1] = im1_fft[up0:down0, up1:down1]

    im2_new = np.abs(np.real(np.fft.ifft2(im2_fft)))
    return im1, im2, im2_new


class SpatialTransform(object):
    '''
    Base class to image transform
    '''

    def __init__(self, image):
        self.image = image
        self.dim = image.GetDimension()

    def apply_transform(self):
        output = []
        out_im = transform_func(self.image, self.image, self.transform, order=1)
        output.append(out_im)
        return output

    def add_transform(self, transform):
        total = sitk.Transform(self.transform)
        total.AddTransform(transform)
        self.transform = total


class AffineTransform(SpatialTransform):
    '''
    Apply random affine transform to input 3D image volume
    '''

    def __init__(self, image, shear_range, scale_range, rot_range, trans_range, flip_prob):
        super(AffineTransform, self).__init__(image)
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.rot_range = rot_range
        self.flip_prob = flip_prob
        self.trans_range = trans_range
        self.transform = sitk.AffineTransform(self.dim)

    def scale(self):
        self.transform = sitk.AffineTransform(self.transform)
        scale = np.eye(self.dim)
        scale = np.diag(1. / np.random.uniform(self.scale_range[0], self.scale_range[1], self.dim))
        matrix = np.array(self.transform.GetMatrix()).reshape((self.dim, self.dim))
        matrix = np.matmul(matrix, scale)
        self.transform.SetMatrix(matrix.ravel())
        self.transform.SetCenter(
            self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize()) / 2.0))

    def rotate(self):
        angles = np.random.uniform(self.rot_range[0], self.rot_range[1], self.dim)
        rads = np.array(angles) / 180. * np.pi
        x_rot = np.eye(self.dim)
        x_rot = [[1., 0., 0.], [0., np.cos(rads[0]), -np.sin(rads[0])], [0., np.sin(rads[0]), np.cos(rads[0])]]
        y_rot = [[np.cos(rads[1]), 0., np.sin(rads[1])], [0., 1., 0.], [-np.sin(rads[1]), 0., np.cos(rads[1])]]
        z_rot = [[np.cos(rads[2]), -np.sin(rads[2]), 0.], [np.sin(rads[2]), np.cos(rads[2]), 0.], [0., 0., 1.]]
        rot_matrix = np.matmul(np.matmul(np.array(x_rot), np.array(y_rot)), np.array(z_rot))
        matrix = np.array(self.transform.GetMatrix()).reshape((self.dim, self.dim))
        matrix = np.matmul(matrix, rot_matrix)
        self.transform = sitk.AffineTransform(self.transform)
        self.transform.SetMatrix(matrix.ravel())
        self.transform.SetCenter(
            self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize()) / 2.0))

    def translate(self):
        global pa_list
        self.transform = sitk.AffineTransform(self.transform)
        params = [random.uniform(self.trans_range[0], self.trans_range[1]),
                  random.uniform(self.trans_range[0], self.trans_range[1])]
        print("Translation: ", params)
        self.transform.SetTranslation(params)
        pa_list = pa_list + params
        # self.transform.SetCenter(self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize())/2.0))

    def shear(self):
        self.transform = sitk.AffineTransform(self.transform)
        axis = np.argsort(np.random.rand(self.dim))
        self.transform.Shear(int(axis[0]), int(axis[1]), np.random.uniform(self.shear_range[0],
                                                                           self.shear_range[1]))
        self.transform.SetCenter(
            self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize()) / 2.0))

    def flip(self):
        flip = np.random.rand(self.dim) > self.flip_prob
        flip_matrix = np.eye(self.dim)
        flip_matrix[np.diag(flip)] = -1.
        self.transform = sitk.AffineTransform(self.transform)
        matrix = np.array(self.transform.GetMatrix()).reshape((self.dim, self.dim))
        matrix = np.matmul(matrix, flip_matrix)
        self.transform.SetMatrix(matrix.ravel())
        self.transform.SetCenter(
            self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize()) / 2.0))

    def affine(self):
        # commented out others since we only need translation for now
        # self.shear()
        # self.rotate()
        self.translate()
        # self.flip()
        # self.scale()

    def apply_transform(self):
        # output = []
        output = transform_func(self.image, self.image, self.transform, order=1)
        # output.append(out_im)
        return output


pa_list = []


def affine_usage(sitk_image):
    '''
    example function to apply affine transform to images
    '''
    global pa_list
    pa_list = []
    params_affine = {
        'scale_range': [0.8, 1.2],
        'rot_range': [-15., 15.],
        'trans_range': [-15., 15.],  # range of translation
        'shear_range': [-0.13, 0.13],
        'flip_prob': 0.3
    }

    affine = AffineTransform(sitk_image, **params_affine)
    affine.affine()
    output = affine.apply_transform()
    print("pa_list: ")
    print(pa_list)
    return output, pa_list
# import numpy as np
# import SimpleITK as sitk
#
#
# def reference_image_build(spacing, size, direction, template_size, dim):
#     # template size: image(array) dimension to resize to: a list of three elements
#     reference_spacing = np.array(size) / np.array(template_size) * np.array(spacing)
#     reference_spacing[0] = 1.2
#     reference_spacing[1] = 1.2
#     reference_image = sitk.Image(template_size, 0)
#     reference_image.SetOrigin(np.zeros(3))
#     reference_image.SetSpacing(reference_spacing)
#     reference_image.SetDirection(direction)
#     return reference_image
#
#
# def centering(img, ref_img, order=1):
#     dimension = img.GetDimension()
#     transform = sitk.AffineTransform(dimension)
#     list1 = img.GetOrigin()
#     list2 = ref_img.GetOrigin()
#     list3 = [None] * 3
#     for i in range(3):
#         list3[i] = list1[i]-list2[i]
#     transform.SetTranslation(list3)
#     # Modify the transformation to align the centers of the original and reference image instead of their origins.
#     centering_transform = sitk.TranslationTransform(dimension)
#     size = img.GetSize()
#     size = [s/2 for s in size]
#     img_center = img.TransformContinuousIndexToPhysicalPoint(size)
#     size_ref = ref_img.GetSize()
#     size_ref = [s / 2 for s in size_ref]
#     reference_center = ref_img.TransformContinuousIndexToPhysicalPoint(size_ref)
#     list1 = transform.GetInverse().TransformPoint(img_center)
#     list2 = reference_center
#     list3 = [None]*3
#     for i in range(3):
#         list3[i] = list1[i]-list2[i]
#     centering_transform.SetOffset(list3)
#     centered_transform = sitk.Transform(transform)
#     centered_transform.AddTransform(centering_transform)
#
#     return transform_func(img, ref_img, centered_transform, order)
#
#
# def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
#     dim = ref_img.GetDimension()
#     affine = sitk.AffineTransform(dim)
#     if target is None:
#         target = np.eye(dim)
#
#     ori = np.reshape(orig_direction, np.eye(dim).shape)
#     target = np.reshape(target, np.eye(dim).shape)
#     affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize()) / 2.0))
#     return transform_func(image, ref_img, affine, order)
#
#
# def transform_func(image, reference_image, transform, order=1):
#     # Output image Origin, Spacing, Size, Direction are taken from the reference
#     # image in this call to Resample
#     if order == 1:
#         interpolator = sitk.sitkLinear
#     elif order == 2:
#         interpolator = sitk.sitkBSpline
#     elif order == 0:
#         interpolator = sitk.sitkNearestNeighbor
#     default_value = 0
#     try:
#         resampled = sitk.Resample(image, reference_image, transform,
#                                   interpolator, default_value)
#     except Exception as e:
#         print(e)
#
#     return resampled
#
#
# def resample_spacing(sitkIm, resolution=0.5, dim=3, template_size=(256, 256), order=1):
#     if type(sitkIm) is str:
#         image = sitk.ReadImage(sitkIm)
#     else:
#         image = sitkIm
#     orig_direction = image.GetDirection()
#     orig_size = np.array(image.GetSize(), dtype=np.int)
#     orig_spacing = np.array(image.GetSpacing())
#     new_size = orig_size * (orig_spacing / np.array(resolution))
#     new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
#     new_size = [int(s) for s in new_size]
#     template_size = (template_size[0], template_size[1], int(orig_size[-1]))
#     ref_img = reference_image_build(resolution, new_size, image.GetDirection(), template_size, dim)
#     centered = centering(image, ref_img, order)
#     transformed = isometric_transform(centered, ref_img, orig_direction, order)
#     return transformed, ref_img
#
#
# def resize_to_size(image, size=(256, 256), order=1):
#     orig_size = np.array(image.GetSize(), dtype=np.int)
#     orig_spacing = np.array(image.GetSpacing())
#     new_size = [int(size[0]), int(size[1]), int(orig_size[-1])]
#     new_spacing = orig_spacing * orig_size / np.array(new_size)
#     if order == 1:
#         interpolator = sitk.sitkLinear
#     elif order == 2:
#         interpolator = sitk.sitkBSpline
#     elif order == 0:
#         interpolator = sitk.sitkNearestNeighbor
#     default_value = 0
#     fltr = sitk.ResampleImageFilter()
#     fltr.SetSize(new_size)
#     fltr.SetOutputSpacing(new_spacing)
#     fltr.SetOutputOrigin(image.GetOrigin())
#     fltr.SetOutputDirection(image.GetDirection())
#     fltr.SetInterpolator(interpolator)
#     image = fltr.Execute(image)
#     return image
#
#
# def resample_scale(sitkIm, ref_img, gt_img=None, scale_factor=1., order=1):
#     sitkIm.SetDirection(np.eye(3).ravel())
#     ref_img.SetDirection(np.eye(3).ravel())
#     gt_img.SetDirection(np.eye(3).ravel())
#     dim = sitkIm.GetDimension()
#     affine = sitk.AffineTransform(dim)
#     scale = np.array(ref_img.GetDirection())
#     scale = np.reshape(scale, (dim, dim))
#     scale[:, 0] *= 1. / scale_factor
#     scale[:, 1] *= 1. / scale_factor
#
#     if gt_img is not None:
#         stats = sitk.LabelShapeStatisticsImageFilter()
#         stats.Execute(sitk.Cast(gt_img, sitk.sitkInt32))
#         center = stats.GetCentroid(1)
#     else:
#         center = sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize()) / 2.0)
#     affine.SetMatrix(scale.ravel())
#     affine.SetCenter(center)
#     transformed = transform_func(sitkIm, ref_img, affine, order)
#     return transformed
#
#
# def swap_labels(labels):
#     unique_label = np.unique(labels)
#     new_label = range(len(unique_label))
#     for i in range(len(unique_label)):
#         label = unique_label[i]
#         newl = new_label[i]
#         labels[labels == label] = newl
#     return labels
#
#
# def swap_labels_back(labels, pred):
#     unique_label = np.unique(labels)
#     new_label = range(len(unique_label))
#     for i in range(len(unique_label)):
#         pred[pred == i] = unique_label[i]
#     return pred
#
#
# def rescale_intensity(slice_im):
#     if type(slice_im) != np.ndarray:
#         raise RuntimeError("Input image is not numpy array")
#     # upper = np.percentile(slice_im, 90)
#     upper = np.percentile(slice_im, 99)
#     lower = np.percentile(slice_im, 20)
#     slice_im[slice_im > upper] = upper
#     slice_im[slice_im < lower] = lower
#     slice_im -= lower
#     rng = upper - lower
#     slice_im = slice_im / rng * 2.
#     slice_im -= 1.
#     # slice_im = (slice_im - np.mean(slice_im))/np.std(slice_im)
#     return slice_im
#
#
# def swap_low_freq(im1, im2, beta):
#     """
#     Change the low frequency of im2 with that of im1
#     Beta: ratio between the swaped region and the image dimension
#     """
#     # im1 = denoise(im1, 10, 0.125)
#     # im2 = denoise(im2, 10, 0.125)
#     im1 = np.squeeze(im1)
#     im2 = np.squeeze(im2)
#     im1 = im1 - np.min(im1)
#     im2 = im2 - np.min(im2)
#     im1_fft = np.fft.fftshift(np.fft.fft2(im1))
#     im2_fft = np.fft.fftshift(np.fft.fft2(im2))
#     change = beta * np.array(im2_fft.shape)
#     up0 = int(im2.shape[0] / 2 - change[0] / 2)
#     down0 = int(im2.shape[0] / 2 + change[0] / 2)
#     up1 = int(im2.shape[1] / 2 - change[1] / 2)
#     down1 = int(im2.shape[1] / 2 + change[1] / 2)
#     # im2_fft[up0:down0, up1:down1] = 0.
#     im2_fft[up0:down0, up1:down1] = im1_fft[up0:down0, up1:down1]
#
#     im2_new = np.abs(np.real(np.fft.ifft2(im2_fft)))
#     return im1, im2, im2_new
