import os, argparse, wandb, torch
import torch.optim as optim

from utils.metrics import *
from utils.dataloader import BrainDataset
from utils.loss import *
from steps.train import train_step
from steps.val import val_step
from unet.unet import UNet

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--restore', help='model path', type=str, default=None)
parser.add_argument('--tag', help='tag', type=str, default='unet')
parser.add_argument('--checkpoint_dir', help='best model', type=str, default="src/best_models")
parser.add_argument('--out_maps_dir', help='outoputs maps dir', type=str, default="src/output_maps")
parser.add_argument('--batch_size', help='Batch size (default: 8)', type=int, default=16)
parser.add_argument('--device', help='Pytorch device', type=str, default="cuda")
parser.add_argument('--data_path', help='Path of images and maps', type=str, default="DeepHealth_IEEE")
parser.add_argument('--map_type', help='Type of map/output', type=str, choices=['NLR_CBV', 'NLR_CBF', 'NLR_TTP', 'NLR_MTT'], default="NLR_CBV")
parser.add_argument('--val_split', help='% val split', type=float, default=0.1)
parser.add_argument('--test_split', help='% test split', type=float, default=0.1)
parser.add_argument('--in_channels', help='num of channels or slice', type=int, default=89)
parser.add_argument('--im_size', help='dim images', type=int, default=512)
parser.add_argument('--resize_dim', help='dim resize images', type=int, default=128)
parser.add_argument('--n_epochs', help='num of epochs', type=int, default=500)
parser.add_argument('--lr', help='learning rate', type=float, default=5e-3)
parser.add_argument('--patience', help='learning patience', type=int, default=10)
parser.add_argument('--wandb_key', help='wandb key api', type=str, default="")
parser.add_argument('--n_blocks', help='num u-net blocks', type=int, default=2)
parser.add_argument('--init_filters', help='num u-net filters', type=int, default=16)
parser.add_argument('--dim_batch_acc', help='size for batch accumulation', type=int, default=512)
parser.add_argument('--pretrained', help='model with weigths', type=bool, default=True)
parser.add_argument('--k_folds', help='number of folds for cross-validation (min=1)', type=int, default=1)

torch.manual_seed(4122015)


def main(args):
	os.environ["WANDB_API_KEY"] = args.wandb_key

	if args.dim_batch_acc<1:
		args.dim_batch_acc = args.batch_size

	args.tag = f'{args.tag}-{args.dim_batch_acc}batch-{args.n_blocks}b-{args.init_filters}init'
	
	if not os.path.exists(os.path.join(args.checkpoint_dir, args.tag)):
		os.makedirs(os.path.join(args.checkpoint_dir, args.tag))

	device = args.device
	
	data_path = args.data_path
	#metadata_path = args.metadata_path
	current_lr = args.lr 
	patience = args.patience

	# TRAINING IMAGE PATHS
	device = torch.device(args.device)

	mse_tot = []
		
	'''
	k-fold cross-validation. If args.k_folds=1 then the cv is not executed

	#TODO: add std to the metrics and modify wanddb log
	'''
	for k in range(args.k_folds):

		print(f'Running fold {k+1}/{args.k_folds}')
		current_lr = args.lr
		print('Building Dataset...')

		train_dataset, val_dataset, test_dataset = BrainDataset(
			data_path,
			args.map_type, 
			num_channels=args.in_channels,
			im_size=args.im_size,
			resize_dim=args.resize_dim,
			val_split=args.val_split,
			k_folds=k
			)
			
		wandb.init(
			project="unito-brain", 
			config=args, 
			tags=[
				f'channels_{args.in_channels}', 
				args.map_type, 
				f'{args.resize_dim}_{args.resize_dim}'
				],
			name=f'{args.map_type}_{args.tag}_{k}',
			reinit=True)
			
		# Load model to device..		
		model = UNet(
			in_channels=args.in_channels,
			out_channels=1,
			n_blocks=args.n_blocks,
			dim=2,
			start_filts=args.init_filters,
			activation='leaky',
			merge_mode='add').to('cuda')
		print(model)
			
		if args.pretrained:
			weights_path = os.path.join(
				args.checkpoint_dir, 
				args.tag, 
				"NLR_CBV",
				f"best_model_{args.map_type}.pt"
				)
			if os.path.exists(weights_path):
				print("Loading CBV weights...")
				model.load_state_dict(torch.load(weights_path))
		
		train_loader = torch.utils.data.DataLoader(
			train_dataset, 
			batch_size=args.batch_size,
			shuffle=True,
			pin_memory=True,
			num_workers=4
		)

		print(
			f"Training dataset loaded [total_len={len(train_loader)} ({(1-args.val_split-args.test_split)*100}%)]"
			)
		val_loader = torch.utils.data.DataLoader(
			val_dataset, 
			batch_size=args.batch_size,
			shuffle=False,
			pin_memory=True,
			num_workers=1
		)
		print(f"Val dataset loaded [total_len={len(val_loader)} ({args.val_split*100}%)]")

		# training setup
		optimizer = optim.Adam(model.parameters(), lr=current_lr)

		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, verbose=True)
		loss_criterion = torch.nn.MSELoss() 
		loss_typ = "MSE"

		loss_ = []

		# start training
		for epoch in range(args.n_epochs):

			if current_lr < 1e-6:
				break

			train_loss, train_psnr, train_cos = train_step(
				args, model, train_loader, loss_criterion, optimizer, device, epoch, k
				)
			val_loss, val_psnr, val_cos = val_step(
				args, model, val_loader, loss_criterion, device, epoch, k
				)
			
			loss_.append(val_loss)

			if val_loss<=min(loss_):
				out_checkpoint = os.path.join(
						args.checkpoint_dir, 
						args.tag, args.map_type)

				if not os.path.exists(out_checkpoint):
					os.makedirs(out_checkpoint)
				
				torch.save(
					model.state_dict(), 
					os.path.join(
						out_checkpoint,
						f"best_model_{args.map_type}_{k}.pt")
						)
				

			lr_scheduler.step(val_loss)

			
			wandb.log(
				{
					f'train_{loss_typ}':train_loss,
					f'val_{loss_typ}':val_loss,
					'train_psnr':train_psnr,
					'val_psnr':val_psnr,
					'train_cosine_similarity':train_cos,
					'val_cosine_similarity':val_cos,
					'lr':current_lr,
					'epoch':epoch
				}
			)
			

			for param_group in optimizer.param_groups:
				current_lr = param_group['lr']

		mse_tot.append(min(loss_))

	print("Saving results to {}".format(os.path.join(
			args.checkpoint_dir, f"{args.map_type}_{args.tag}_results.txt")))
	with open(
		os.path.join(
			args.checkpoint_dir, f"{args.map_type}_{args.tag}_results.txt"), 
		'a'
		) as f:
		f.write(f'MSE: {np.mean(mse_tot)}, \sd(MSE): {np.std(mse_tot)}')


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
