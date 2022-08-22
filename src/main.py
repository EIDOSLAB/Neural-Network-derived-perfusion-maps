import torch
import torch.optim as optim
import torch.backends.cudnn

import os
import torchvision
from torchvision import transforms

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

beta = 1.0
def train(model, train_loader, loss_criterion, optimizer, device, epoch):
	model.train()
	tot_loss = 0.
	n_batch = 0.
	optimizer.zero_grad()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(device)
		target = target.to(device)
		outputs = model(data)
		outputs = torch.sigmoid(outputs)
		loss = loss_criterion(outputs, target)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		tot_loss += loss.item()
		n_batch += 1
	print('Epoch {}   Train Loss: {}'.format(epoch, tot_loss / n_batch))

def test(model, train_loader, loss_criterion, device, epoch, save_path=None):
	model.eval()
	tot_loss = 0.
	n_batch = 0.
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):
			data = data.to(device)
			target = target.to(device)
			outputs = model(data)
			outputs = torch.sigmoid(outputs)
			loss = loss_criterion(outputs, target)
			tot_loss += loss.item()
			n_batch += 1
			if (save_path != None) and batch_idx == 0:
				#print(outputs.max(), outputs.min())
				a = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(outputs).cpu())
				b = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(target).cpu())
				a.save(save_path+'_'+str(epoch)+'.png')
				b.save(save_path+'_'+str(epoch)+'_GT.png')
		print('Epoch {}   Test Loss: {}'.format(epoch, tot_loss / n_batch))
	return tot_loss



parser = argparse.ArgumentParser()
parser.add_argument('--restore', help='model path', type=str, default=None)
parser.add_argument('--batch_size', help='Batch size (default: 8)', type=int, default=8)
parser.add_argument('--device', help='Pytorch device', type=str, default="cuda:0")

def main():
	args = parser.parse_args()

	# parameters setting
	num_epochs = 250
	last_epoch = 0
	n_classes = 1 

	########################################################################################################################
	# DATA LOADING
	########################################################################################################################

	print('Building Dataset....')

	# Define input training data and labels directories

	# TRAINING IMAGE PATHS
	device = torch.device(args.device)
	torch.cuda.set_device(device)
	loss_criterion = torch.nn.MSELoss()
	for in_channels in [89]:
		# Load model to device..
		print('Initializing model...')
		model = model = UNet(in_channels=in_channels, out_channels=n_classes, init_features = 64).to(device)
		print('...model has been initialized')

		for sizeimage in [512]:
			#paths eventually to be set properly
			train_data_path = os.path.normpath('/home/ubuntu/data/DH3/'+str(sizeimage)+'/input_tensored')
			test_data_path = os.path.normpath('/home/ubuntu/data/DH3/'+str(sizeimage)+'/validate')
			train_labels_path = os.path.normpath('/home/ubuntu/data/DH3/'+str(sizeimage)+'/CBV')#targets-in this case, CBV

	##############################################################s##########################################################
	# SETUP NEURAL NETWORK, LOSS FUNCTION
	########################################################################################################################

			train_dataloader_input = customDataLoader(train_data_path, train_labels_path, in_channels)
			test_dataloader_input = customDataLoader(test_data_path, train_labels_path, in_channels)
			train_loader = torch.utils.data.DataLoader(dataset=train_dataloader_input,
                                                                   batch_size=args.batch_size,
			                                           shuffle=True,
			                                           pin_memory=True,
			                                           num_workers=1) #

			test_loader = torch.utils.data.DataLoader(dataset=test_dataloader_input,
			                                           batch_size=args.batch_size,
			                                           shuffle=False,
			                                           pin_memory=True,
			                                           num_workers=1) #
			print('....Dataset built!')

			optimizer = optim.Adam(model.parameters(), lr=5e-5)
			lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

			print('\n===============================TRAINING T={} size {}x{} BEGINS==============================='.format(in_channels, sizeimage, sizeimage))
			epoch = 0
			current_lr = 5e-3
			while current_lr > 1e-6:
				train(model, train_loader, loss_criterion, optimizer, device, epoch)
				valid_loss = test(model, test_loader, loss_criterion, device, epoch)
				lr_scheduler.step(valid_loss)
				epoch += 1
				for param_group in optimizer.param_groups:
					current_lr = param_group['lr']
		print('\n===============================TRAINING COMPLETE=============================')
		torch.save(model, 'models/this_model.pt')
if __name__ == '__main__':
    main()
