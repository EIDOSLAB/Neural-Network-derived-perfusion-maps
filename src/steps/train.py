
import torch
import numpy as np
from utils.preproc import normalization
from utils.utils import plt_map
from torchmetrics import PeakSignalNoiseRatio as PSNR
import torchio as tio
import random
import torch.nn.functional as F
from tqdm import tqdm



def train_step(args, model, train_loader, loss_criterion, optimizer, device, epoch, fold):

    model.train()
    psnr = PSNR()
    cosine = torch.nn.CosineSimilarity(dim=0)

    tot_loss = 0.
    tot_psnr = 0.
    tot_cosine = 0.
    n_batch = 0.

    accum_iter = args.dim_batch_acc

    for i, (data, target, p) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epochs}')):

        optimizer.zero_grad()

        #spacing = int(np.floor(float(data.size(1))/(args.in_channels)))
        #channels = np.arange(0, data.size(1), spacing)[:args.in_channels]
        #data = data[:, channels, :, :]

        '''
        n, d, w, h = data.shape
        data = data.view(n, 1, w, h, d)

        #fill_ = torch.cat([torch.mean(data, dim=-1, keepdim=True)]*7, dim=-1)
        fill_ = torch.zeros((n, 1, w, h, 7))
        data = torch.cat((data, fill_), -1)
        '''
        #data, target = anomaly_detect(data, target)
        #

        batch_dim, time, w, h = data.shape

        #down = torch.nn.Upsample(size=(89, 256, 256))
        #down_target = torch.nn.Upsample(size=(1, 256, 256))
        #up = torch.nn.Upsample(size=(1, 512, 512))
        # batch, 89, 8, 512, 512
        #data = data.view(batch_dim, time, cross_section, w, h)
        #target = target.view(batch_dim, 1, cross_section, w, h)

        data = data.view(batch_dim, time, w, h)
        target = target.view(batch_dim, 1, w, h)

        #data = _transform(data)
        data = data.to('cuda')#to(device)
        target = target.to('cuda')

        #data = down(data)
        #target = down_target(target)

        with torch.set_grad_enabled(True):
            #outputs = torch.mean(model(data), dim=2).view(batch_dim, 1, 1, w, h)

            outputs = model(data)
            #scale = torch.nn.Upsample(size=(1, outputs.shape[-1], outputs.shape[-1]))
            #target = scale(target)

            #outputs = F.normalize(outputs, dim = 0)
            #outputs = torch.sigmoid(outputs)
            #outputs = torch.max(outputs, dim=-1)

            if (epoch%10 == 0 or epoch==0) and i%10==0:
                with torch.no_grad():
                    for k in range(batch_dim):
                        plt_map(
                            args, target[k, 0].cpu(), 
                            outputs[k, 0].cpu(), 
                            "train", 
                            f"{p[0]}", 
                            epoch, 
                            k,
                            fold
                        )

            loss = loss_criterion(outputs, target)
            loss.backward()

            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            #optimizer.step()

            psnr_metric = psnr(outputs.cpu(), target.cpu())
            cos_sim = torch.mean(cosine(outputs.cpu(), target.cpu()))

            tot_loss += loss.item()
            tot_psnr += psnr_metric.item()
            tot_cosine += cos_sim.item()

            n_batch += 1

    loss_out = tot_loss/n_batch
    psnr_out = tot_psnr/n_batch
    cos_out = tot_cosine/n_batch

    print('Epoch {} Train Loss: {}'.format(epoch, round(loss_out, 4)))

    return loss_out, psnr_out, cos_out



