import torch
import torch.optim as optim
import torch.backends.cudnn

from unet import *
from metrics import *
from dataloader import customDataLoader
from torchvision import transforms
import argparse
import numpy as np

def infer(model, input_tensor, device):
	model.eval()
	with torch.no_grad():
		input_tensor = (input_tensor.type(torch.float32)) / 255.0
		data = (input_tensor.unsqueeze(0)).to(device)
		output = model(data).cpu()
		return output


parser = argparse.ArgumentParser()
parser.add_argument('--restore', help='model path', type=str, default='/home/tarta/uc3-brain/src/models/370.pt')
parser.add_argument('--device', help='Pytorch device (default: cuda:0)', type=str, default="cuda:0")

def main():
    args = parser.parse_args()

    in_channels = 89# data per pixel
    n_classes = 1 # need to further set correctly ...

    print('Building Dataset....')

    # Define input training data and labels directories

    # TRAINING IMAGE PATHS
    infer_tensor_path = '/home/tarta/data/DH3/128/validate_128/MOL-002-00001.pt'

    print('....Dataset built!')

    print('Initializing model...')

    device = torch.device(args.device)
    #torch.cuda.set_device(0)
    # Load model to device..
    #model = UNet(in_channels=in_channels).to(device)
    print('...model has been initialized')

    model = torch.load(args.restore, map_location=device)
    in_tensor = torch.load(infer_tensor_path)

    outcome = infer(model, in_tensor, device)[0,0,:,:]
    im = transforms.ToPILImage()(outcome)
    im.save('prova.png')

if __name__ == '__main__':
    main()
