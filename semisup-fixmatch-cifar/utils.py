'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

import numpy as np  # For createImbIdxs() & make_imb_data()

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#%%
def createImbIdxs(labels, n_data_per_class) :
    '''
    Creates a List containing Indexes of the Imbalanced Classification

    Input: 
        labels: Ground Truth of Dataset
        n_data_per_class: Class Distribution of Dataset desired

    Output:
        data_idxs: List containing indexes for Dataset 
    '''
    labels = np.array(labels) # Classification Ground Truth 
    data_idxs = []  # Collect Ground Truth Indexes

    for i in range( len(n_data_per_class) ) :
        idxs = np.where(labels == i)[0]
        data_idxs.extend(idxs[ :n_data_per_class[i] ])

    return data_idxs

def checkReverseDistb(imb_ratio) :
    reverse = False
    if imb_ratio / abs(imb_ratio) == -1 :
        reverse = True
        imb_ratio = imb_ratio * -1

    return reverse, imb_ratio

def make_imb_data(max_num, class_num, gamma):
    reverse, gamma = checkReverseDistb(gamma)

    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if reverse :
        class_num_list.reverse()
    print(class_num_list)
    return list(class_num_list)
#%%
