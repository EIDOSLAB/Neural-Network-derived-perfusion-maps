import torch
import torch.optim as optim
import torch.backends.cudnn

import os
import torchvision

from metrics import *
from dataloader import customDataLoader
from loss import *
from unet import *

import argparse

import json
import os
import argparse
import multiprocessing
import importlib
import time

import random
import numpy as np

import torch
import torchvision
from dicomutils import pngtodicom,tensortodicom, handler_graydicomtoheatmapdicom, handler_graypngtoheatmapdicom
from glob import glob
from dicomutils import grayscaletoheatmap

beta = 1.0

def test(model, train_loader, loss_criterion, device, epoch, print_flag=None):
	model.eval()
	tot_loss = 0.
	n_batch = 0.
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):
			#print(batch_idx)
			data = data.to(device)
			target = target.to(device)
			outputs = model(data)*beta
			outputs = torch.sigmoid(outputs)
			#print(data.size())
			#print(outputs.size())
			loss = loss_criterion(outputs, target)
			tot_loss += loss.item()
			n_batch += 1
			if (batch_idx == 0) and (print_flag!= None):
				#outputs = torch.sigmoid(outputs)
				#print(outputs.max(), outputs.min())
				a = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(outputs).cpu())

				b = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(target).cpu())

				a.save('sample_out/'+print_flag+'.png')
				b.save('sample_out/GT_'+print_flag+'.png')
		print('Epoch {}   Test Loss: {}'.format(epoch, tot_loss / n_batch))
	return tot_loss


parser = argparse.ArgumentParser()
parser.add_argument('--restore', help='model path', type=str, default=None)
parser.add_argument('--batch_size', help='Batch size (default: 8)', type=int, default=8)
parser.add_argument('--device', help='Pytorch device (default: cuda:0)', type=str, default="cuda:1")
dicom_save = True
GT_save = True


def main():
	type="CBV"
	device = torch.device('cuda:1')
	torch.cuda.set_device(device)
        torch.nn.Module.dump_patches = True

	os.system('mkdir /home/tarta/Desktop/'+type)
	input_tensors = glob("/home/tarta/data/DH3/512/validate/*")
	##create folders
	folder_BW = "/home/tarta/Desktop/"+type+"/BW"
	folder_COLOR = "/home/tarta/Desktop/"+type+"/COLOR"
	folder_GT = "/home/tarta/Desktop/"+type+"/GROUND_TRUTH"
	folder_GT_COLOR = "/home/tarta/Desktop/"+type+"/GROUND_TRUTH_COLOR"
	os.system('mkdir '+folder_BW)
	os.system('mkdir '+folder_COLOR)
	if GT_save:
		os.system('mkdir '+folder_GT)
		os.system('mkdir '+folder_GT_COLOR)
	for network in [5,7,10,18,27,44,89]:
		os.system('mkdir '+folder_BW+'/'+str(network))
		os.system('mkdir '+folder_COLOR+'/'+str(network))
	for i in input_tensors:
		my_i = (i.split('/'))[-1]
		###ground_truth
		target_dicom = '/media/DH_UC3_Brain/processed_data_Benninck/'+my_i[:7]+'_Registered_Filtered_3mm_20HU_Maps/NLR_'+type+'/NLR_'+type+my_i[8:13]+'.dcm'

		save_BW_dicom = folder_GT+'/'+my_i[:13]+'.dcm'
		save_COLOR_dicom = folder_GT_COLOR+'/'+my_i[:13]+'.dcm'
		if dicom_save:
			os.system("cp "+str(target_dicom) +" "+save_BW_dicom)
			handler_graydicomtoheatmapdicom(save_BW_dicom, save_COLOR_dicom)
		for network in [5,7,10,18,27,44,89]:
                        model = torch.load('models/'+type+"_"+str(network)+".pt", map_location=device)
			output_name = folder_BW+'/'+str(network)+'/'+my_i[:13]+'.dcm'
			output_name_COLOR = folder_COLOR+'/'+str(network)+'/'+my_i[:13]+'.dcm'
			print(network, my_i[:13])
			main_sec(model, device, i, target_dicom, output_name,output_name_COLOR, network,type)

def main_sec(model, device, input_tensor, target_dicom, output_name, output_name_COLOR, network,type):
	args = parser.parse_args()

	# parameters setting
	#input_tensor = '/home/tarta/data/DH3/512/validate/MOL-002-00008.pt'
	#target_dicom = '/media/tarta/Data/DH_UC3_Brain/processed_data_Benninck/MOL-002_Registered_Filtered_3mm_20HU_Maps/NLR_CBV/NLR_CBV00008.dcm'
	#output_name = '/home/tarta/Desktop/NN_CBV00008.dcm'

	#device = torch.device(args.device)
	#torch.cuda.set_device(device)

	if args.restore is not None:
		#TODO: fix -> save/load checkpoint {model.state_dict(), epochs} instead of model
		model = torch.load(args.restore, map_location=device)
		last_epoch = 50
		print('Restoring training of {args.restore} from epoch {last_epoch}')


	########################################################################################################################
	# PERFORM NETWORK TRAINING AND SAVE LOSS DATA
	########################################################################################################################

	# TRAINING IMAGE PATHS
	infer_sample = os.path.normpath(input_tensor)
	im_frame = (((torch.load(infer_sample))).type(torch.float32)) #/ 255.0
	img_frame = im_frame.unsqueeze(dim = 1)
	#a = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(img_frame).cpu())
	#a.save('all_samples.png')
	#model = torch.load('models/'+type+'_RESCALEfull'+str(network)+'.pt', map_location=device)
	model.eval()
	with torch.no_grad():
		if im_frame.size()[0] >= network:
			spacing = int(np.floor(float(im_frame.size()[0])/(network)))
			channels = np.arange(0, im_frame.size()[0], spacing)[0:network]
			data = (im_frame[channels,:,:].to(device)-0.2375)/0.3159
			data = data.unsqueeze(dim = 0)
			#data = data.unsqueeze(dim=1)
			outputs = model(data)
			print(outputs.shape)
			outputs = (((torch.sigmoid(outputs)).squeeze(dim = 0)).squeeze(dim=1)).squeeze(dim=2)
			#outputs = torch.argmax(outputs, 1).type(torch.float)/3.0
			#tensortodicom(target_dicom, outputs.cpu(), output_name)
			a = torchvision.transforms.ToPILImage()(outputs.cpu())##torchvision.transforms.ToPILImage()
			png_name = output_name.replace('.dcm', '.png')
			a.save(png_name)
			grayscaletoheatmap(png_name, png_name.replace('BW', 'COLOR'))
			if dicom_save:
				pngtodicom(target_dicom, a, output_name, network)
				handler_graypngtoheatmapdicom(output_name, output_name_COLOR)

if __name__ == '__main__':
    main()
