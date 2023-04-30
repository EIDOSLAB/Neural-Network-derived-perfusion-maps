import os, json, random
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchio as tio

from utils.utils import *
from utils.preproc import standardization, normalization
torch.manual_seed(42)


# customized Data loader
class BrainDataset(Dataset):
	def __init__(
		self, 
		img_directory, 
		map_type, 
		num_channels, 
		im_size, 
		resize_dim,
		dataset_type, 
		val_test_split, 
		k_folds,
		transformations=None
		): #READ DATA

		super(BrainDataset, self).__init__()
		self.img_directory = img_directory
		self.num_channels = num_channels
		self.transformations = transformations
		self.map_type = map_type
		self.im_size = im_size
		self.resize_dim = resize_dim
		self.dataset_type = dataset_type
		self.val_split, self.test_split = val_test_split
		
		self.k_folds = k_folds

		self.geom_transform = geom_transform(self)

		# load or generate metadata from DeepHealth_IEEE
		data = get_metadata(self)

		# split data based on type of map, and number of patient for train/val/test
		self.images_list, self.maps_list, self.patients_list = split_data_cv(
			(
				data["images"], 
				data["maps"][self.map_type], 
				data["patients"]), 
			self.dataset_type,
			self.val_split,
			self.test_split,
			self.k_folds
			)

		# import images (volumes) and maps from dicom. 
		self.images, self.maps, self.patients = get_imgs_maps(
			self,
			self.images_list, 
			self.maps_list, 
			self.patients_list, 
			im_size
		)


	def __getitem__(self, index): # RETURN ONE ITEM ON THE INDEX
		#self.images[index] = remove_noise(self.images[index])
		# standardization of volumes with mean and var; normalization of maps btw 0 and 1
		this_img = torch.load(self.images[index])
		this_mp = torch.load(self.maps[index])
		this_img = standardization(this_img) #standardization
		this_img = normalization(this_img) #standardization

		return this_img, this_mp, self.patients[index]

	def __len__(self): # RETURN THE DATA LENGTH
		return len(self.images)



def geom_transform(self):
	t = [T.Resize((self.resize_dim, self.resize_dim))]
	
	#if self.dataset_type=='train':
	#	t+=[
	#		T.RandomHorizontalFlip(0.3),
	#		T.RandomVerticalFlip(0.3),
	#		T.RandomRotation(15)
	#		]
	return T.Compose(t)
