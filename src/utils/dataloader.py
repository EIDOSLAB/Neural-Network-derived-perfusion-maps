import os, json, random
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchio as tio

from utils.utils import *
from utils.preproc import standardization, normalization
torch.manual_seed(42)


# customized Data loader
def BrainDataset(
		img_directory, 
		map_type, 
		num_channels, 
		im_size, 
		resize_dim,
		val_split, 
		k_folds
		): #READ DATA

		# load or generate metadata from DeepHealth_IEEE
		data = get_metadata(img_directory)

		# split data based on type of map, and number of patient for train/val/test
		images_list, maps_list, patients_list = split_data_cv(
			(
				data["images"], 
				data["maps"][map_type], 
				data["patients"]), 
			'train',
			val_split,
			k_folds
			)

		# import images (volumes) and maps from dicom. 
		images, maps, patients = get_imgs_maps(
			images_list, 
			maps_list, 
			patients_list, 
			im_size, img_directory, map_type
		)
		mean = 0
		total_items = 0
		for this_image in tqdm(images):
			this_image = torch.load(this_image)
			total_items += this_image.numel()
			mean += torch.sum(this_image)

		mean /= total_items
		print("Average: ", mean)
		std = 0
		for this_image in tqdm(images):
			this_image = torch.load(this_image)
			std += torch.sum((this_image - mean)**2)
		std /= total_items
		std = torch.sqrt(std)
		print("Std: ", std)

		train_dataset = BrainDataset_internal(images, maps, patients, mean, std)
		images_list, maps_list, patients_list = split_data_cv(
			(
				data["images"], 
				data["maps"][map_type], 
				data["patients"]), 
			'val',
			val_split,
			k_folds
			)

		# import images (volumes) and maps from dicom. 
		images, maps, patients = get_imgs_maps(
			images_list, 
			maps_list, 
			patients_list, 
			im_size, img_directory, map_type
		)
		val_dataset = BrainDataset_internal(images, maps, patients, mean, std)

		images_list, maps_list, patients_list = split_data_cv(
			(
				data["images"], 
				data["maps"][map_type], 
				data["patients"]), 
			'test',
			val_split,
			k_folds
			)

		# import images (volumes) and maps from dicom. 
		images, maps, patients = get_imgs_maps(
			images_list, 
			maps_list, 
			patients_list, 
			im_size, img_directory, map_type
		)
		test_dataset = BrainDataset_internal(images, maps, patients, mean, std)

		return train_dataset, val_dataset, test_dataset

class BrainDataset_internal(Dataset):
	def __init__(
		self, 
		images, 
		maps, 
		patients, 
		mean, 
		std
		):
		super(BrainDataset_internal, self).__init__()
		self.images = images
		self.maps = maps
		self.patients = patients 
		self.mean = mean
		self.std = std

	def __getitem__(self, index): # RETURN ONE ITEM ON THE INDEX
		#self.images[index] = remove_noise(self.images[index])
		# standardization of volumes with mean and var; normalization of maps btw 0 and 1
		this_img = torch.load(self.images[index])
		this_mp = torch.load(self.maps[index])
		this_img = (this_img - self.mean) / self.std#standardization(this_img) #standardization
		#this_img = normalization(this_img) #standardization

		return this_img, this_mp, self.patients[index]

	def __len__(self): # RETURN THE DATA LENGTH
		return len(self.images)