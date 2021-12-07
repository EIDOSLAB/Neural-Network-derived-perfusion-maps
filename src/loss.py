import torch.nn as nn
import torch
import torch.nn.functional as F


def mixed(input, target):
	alpha = 10.
	loss = alpha * focal(input, target) - torch.log(dice(input, target))
	return loss.mean()

def dice(input, target):
	input = torch.sigmoid(input)
	smooth = 1.0
	iflat = input.view(-1)
	tflat = target.view(-1)
	intersection = (iflat * tflat).sum()
	return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def diceloss(input, target):
	input = torch.sigmoid(input)
	smooth = 1.0
	iflat = input.view(-1)
	tflat = target.view(-1)
	intersection = (iflat * tflat).sum()
	return 1. - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def focal(input, target):
	gamma = 2.
	max_val = (-input).clamp(min=0)
	loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
	invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
	loss = (invprobs * gamma).exp() * loss
	return loss.mean()
