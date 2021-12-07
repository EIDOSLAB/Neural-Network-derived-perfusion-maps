import os
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
#from PIL import Image
import torch
import numpy as np

# customized Data loader
# We recommend users firstly convert DICOM to JPEG, then load JPEG images in dataloader
class customDataLoader(Dataset):
	def __init__(self, img_directory, mask_directory, num_channels, transformations=None): #READ DATA
		self.img_directory = img_directory
		####for the moment, we just use MOL exams
		self.img_list = []
		for element in os.listdir(img_directory):
			if ('MOL' in element) and not ('60' in element) and not ('MOL-001' in element) and not ('MOL-062' in element)and not ('MOL-063' in element)and not ('MOL-061' in element):
				self.img_list.append(element)
		self.mask_directory = mask_directory
		self.num_channels = num_channels
		self.transformations = transformations
		#if mask_directory == None:
		#	self.mask_list = None
		#else:
		#self.mask_list = os.listdir(mask_directory)
		#self.seg_list = pd.read_csv(seg_list, header = None, squeeze = True).set_index(0)[1].to_dict()

	def __getitem__(self, index): # RETURN ONE ITEM ON THE INDEX
		#print(self.img_directory + "/" + self.img_list[index])
		#print(self.mask_directory + "/" + self.img_list[index][0:-3]+'.png')
		im_frame = ((((torch.load(self.img_directory + "/" + self.img_list[index]))).type(torch.float32)))
		spacing = int(np.floor(float(im_frame.size()[0])/(self.num_channels)))
		channels = np.arange(0, im_frame.size()[0], spacing)[0:(self.num_channels)]
		im_frame = (im_frame[channels,:,:]- 0.2375)/0.3159
		#get_seg = self.seg_list[self.img_list[index][:-4]]
		#if (self.mask_directory == None):
		#	return im_frame, self.img_list[index][:-4]
		#else:
		mask_frame = (((torch.load(self.mask_directory + "/" + self.img_list[index]))).type(torch.float32)).unsqueeze(dim = 0)
		#mask_frame = mask.type(torch.float32)#(mask > 0.5).type(torch.float32)
		return im_frame, mask_frame

	def __len__(self): # RETURN THE DATA LENGTH
		return len(self.img_list)
