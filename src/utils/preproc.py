import torch
import pydicom as dicom
import numpy as np
from skimage import morphology
from skimage.transform import resize
from scipy import ndimage


def to_grayscale(image):
    if image.max()!=0:
        image = image.astype(float)
        return np.uint8(np.maximum(image,0)*255.0/image.max())
    else:
        print('[warning]: empty frame.')
        return np.uint8(image.astype(float))

def standardization(tensor):
	tensor -= torch.mean(tensor)
	tensor /= torch.std(tensor)
	return tensor 

def normalization(img):
    img -= torch.min(img)
    img /= (torch.max(img)-torch.min(img))
    return img

def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    image = image*slope 
    image += intercept
    return image

def window_image(window_image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def dicom_to_tensor(file_path, im_size):
    dicom_image = dicom.read_file(file_path)
    image = dicom_image.pixel_array
    center = dicom_image.WindowCenter
    width = dicom_image.WindowCenter
    #h = str(dicom_image.ImagePositionPatient[-1])
    image = transform_to_hu(
        dicom_image, 
        image
        )
    
    image = window_image(
                image,
                center, 
                width
                )
    #image = remove_noise(image)
    image = resize(image, (im_size, im_size), anti_aliasing=True)
    return torch.tensor(image) #, str(h)


def remove_noise(brain_image):
   # brain_image = brain_image.numpy()
    brain_image = brain_image
    maped_image = np.zeros(brain_image.shape)
    frame = to_grayscale(brain_image)
    segmentation = morphology.dilation(frame, np.ones((1, 1)))
    labels, _ = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(np.int64))
    label_count[0] = 0
    map = labels == label_count.argmax()
    map = morphology.dilation(map, np.ones((1, 1)))
    map = ndimage.binary_fill_holes(map)
    map = morphology.dilation(map, np.ones((3, 3)))
    maped_image = map*frame
    # for i, frame in enumerate(brain_image):
    #     frame = to_grayscale(frame)
    #     segmentation = morphology.dilation(frame, np.ones((1, 1)))
    #     labels, _ = ndimage.label(segmentation)
    #     label_count = np.bincount(labels.ravel().astype(np.int64))
    #     label_count[0] = 0
    #     map = labels == label_count.argmax()
    #     map = morphology.dilation(map, np.ones((1, 1)))
    #     map = ndimage.binary_fill_holes(map)
    #     map = morphology.dilation(map, np.ones((3, 3)))
    #     maped_image[i] = map*frame
    return torch.from_numpy(maped_image).to(torch.float)


def create_tensor(_list, _size):

    t = torch.zeros((len(_list), _size, _size))
    for i, _path in enumerate(_list):
        frame = dicom_to_tensor(_path, _size)
        t[i] = frame
    return t
