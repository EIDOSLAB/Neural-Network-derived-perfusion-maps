import torch, random
import numpy as np
from utils.preproc import normalization
from utils.utils import plt_map
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import PearsonCorrCoef
import torch.nn.functional as F

#torch.cuda.set_device(1)

def val_step(args, model, val_loader, loss_criterion, device, epoch, fold, save_path=None):
    model.eval()
    psnr = PSNR()
    cosine = torch.nn.CosineSimilarity(dim=0)

    up = torch.nn.Upsample(size=(1, 512, 512))

    tot_cosine = 0.
    tot_loss = 0.
    tot_psnr = 0.
    n_batch = 0.
    #n = random.randint(0, len(val_loader))
    with torch.no_grad():
        for i, (data, target, p) in enumerate(val_loader):

            #spacing = int(np.floor(float(data.size(1))/(args.in_channels)))
            #channels = np.arange(0, data.size(1), spacing)[:args.in_channels]
            #data = data[:, channels, :, :]

            '''
            n, d, w, h = data.shape
            data = data.view(n, 1, w, h, d)

            fill_ = torch.zeros((n, 1, w, h, 7))
            #fill_ = torch.cat([torch.mean(data, dim=-1, keepdim=True)]*7, dim=-1)
            data = torch.cat((data, fill_), -1)
            '''
            batch_dim, time, w, h = data.shape
            #down = torch.nn.Upsample(size=(89, 256, 256))
            #down_target = torch.nn.Upsample(size=(1, 256, 256))
            #up = torch.nn.Upsample(size=(1, 512, 512))
            # batch, 89, 8, 512, 512
            #data = data.view(batch_dim, time, cross_section, w, h)
            #target = target.view(batch_dim, 1, cross_section, w, h)

            data = data.view(batch_dim, time, w, h)
            target = target.view(batch_dim, 1, w, h)

            data = data.to('cuda')#to(device)
            target = target.to('cuda')

            #data = down(data)
            #target = down_target(target)
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
                            "val", 
                            f"{p[0]}", 
                            epoch, 
                            k,
                            fold
                        )

            loss = loss_criterion(outputs, target)
            psnr_metric = psnr(outputs.cpu(), target.cpu())
            cos_sim = torch.mean(cosine(outputs.cpu(), target.cpu()))


            tot_loss += loss.item()
            tot_psnr += psnr_metric.item()
            tot_cosine += cos_sim.item()
            n_batch += 1

        loss_out = tot_loss/n_batch
        psnr_out = tot_psnr/n_batch
        cos_out = tot_cosine/n_batch
        print('Epoch {}   Val Loss: {}'.format(epoch, round(loss_out, 4)))

    return loss_out, psnr_out, cos_out

