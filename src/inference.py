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

beta = 1.0

def train(model, train_loader, loss_criterion, optimizer, device, epoch):
	model.train()
	tot_loss = 0.
	n_batch = 0.
	optimizer.zero_grad()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(device)
		target = target.to(device)
		outputs = model(data)*beta
		#print(data.size())
		#print(outputs.size())
		outputs = torch.sigmoid(outputs)
		loss = loss_criterion(outputs, target)
		loss.backward()
		# Update weights
		optimizer.step()
		optimizer.zero_grad()
		tot_loss += loss.item()
		n_batch += 1
	print('Epoch {}   Train Loss: {}'.format(epoch, tot_loss / n_batch))
		#print(batch_idx, loss.item())
		# total_loss = get_loss_train(model, data_train, criterion)

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
parser.add_argument('--device', help='Pytorch device (default: cuda:1)', type=str, default="cuda:1")

def main():
	args = parser.parse_args()

	# parameters setting
	n_classes = 2 # need to further set correctly ...

	########################################################################################################################
	# DATA LOADING
	########################################################################################################################


	##############################################################s##########################################################
	# SETUP NEURAL NETWORK, LOSS FUNCTION
	########################################################################################################################

	device = torch.device(args.device)
	torch.cuda.set_device(device)

	if args.restore is not None:
		#TODO: fix -> save/load checkpoint {model.state_dict(), epochs} instead of model
		model = torch.load(args.restore, map_location=device)
		last_epoch = 50
		print('Restoring training of {args.restore} from epoch {last_epoch}')


	########################################################################################################################
	# PERFORM NETWORK TRAINING AND SAVE LOSS DATA
	########################################################################################################################

	# TRAINING IMAGE PATHS
	infer_sample = os.path.normpath('/home/tarta/data/DH3/512/validate/MOL-002-00008.pt')
	im_frame = (((torch.load(infer_sample))).type(torch.float32)) / 255.0
	img_frame = im_frame.unsqueeze(dim = 1)
	a = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(img_frame).cpu())
	a.save('all_samples.png')
	for channels in [1,2,3]:
		model = torch.load('models/trained_model_'+str(channels)+'.pt', map_location=device)
		model.eval()
		with torch.no_grad():

			spacing = int(np.floor(float(im_frame.size()[0])/(channels)))
			channels = np.arange(0, im_frame.size()[0], spacing)[0:channels]
			data = im_frame[channels,:,:].to(device)
			data = data.unsqueeze(dim = 0)
			outputs = model(data)*beta
			outputs = torch.sigmoid(outputs)
			a = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(outputs).cpu())
			a.save('inferred'+str(channels)+'.png')

if __name__ == '__main__':
    main()
