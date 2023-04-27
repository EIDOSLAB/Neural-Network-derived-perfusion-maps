import os
import torch
import numpy as np

from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
                This should be differentiable.
                            pred: tensor with first dimension as batch
                                           target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def accuracy_check(map, prediction):
    ims = [map, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())

def accuracy_check_for_batch(maps, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(maps[index], predictions[index])
    return total_acc/batch_size

def get_loss(model, train_loader, loss_criterion, device):
  """
      Calculate loss
  """
  model.eval()
  total_acc = 0
  total_loss = 0
  batch = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.to(device)
    with torch.no_grad():
      outputs = model(data)
      if isinstance(outputs,tuple):
        outputs,z = outputs
      loss = loss_criterion(outputs, target)
      preds = torch.argmax(outputs, dim=1).float()
      acc = accuracy_check_for_batch(target.cpu(), preds.cpu(), data.size()[0])
      total_acc = total_acc + acc
      total_loss = total_loss + loss.cpu().item()
      batch += 1
  return total_acc/(batch), total_loss/(batch)

def eval_net(net, dataset, device):
	"""Evaluation without the densecrf with the dice coefficient"""
	net.eval()
	tot = 0
	for batch_idx, (data, target) in enumerate(dataset):

		data = data.to(device)
		target = target.to(device)

		torch.cuda.empty_cache()
		map_pred = net(data)
		map_pred = (map_pred > 0.5).float()

		tot += dice_loss(map_pred, target).item()

	return tot / (batch_idx + 1)
