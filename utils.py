import os
import torch
import pickle
import numpy as np
import pandas as pd

# ==================== General utils ====================#
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
def makepath(path):
    makedir(os.path.dirname(path))
    
def save_predictions(preds, pred_path):
    makepath(pred_path) 
    with open(pred_path,'wb') as f: pickle.dump(preds, f)
        
def save_checkpoint(state, checkpoint_path):
    makepath(checkpoint_path)
    torch.save(state, checkpoint_path)
    
def record_info(info, filename):
    print(''.join(['{}: {}   '.format(k,v) for k,v in info.items()]))
    
    df = pd.DataFrame.from_dict(info)
    column_names = list(info.keys())
    
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names) 
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0] if len(topk) == 1 else res
               
class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        
# image utils
def tensor2img(tensor):
    return tensor.permute(1,2,0).cpu().detach().numpy()

def img2uint8(array):
    return (array * 255).astype(np.uint8)

def tensor2uint8(tensor):
    if tensor.shape[0] == 3:
        return img2uint8(tensor2img(tensor))
    else:
        img = img2uint8(tensor2img(tensor))[:,:,0]
        return np.stack([img,img,img], axis=2)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# from PIL import Image
# import numpy as np

# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.nn as nn
# import torch
# import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# from torchvision.utils import make_grid

# import cv2
# from IPython.display import HTML

# # image utils
# def overlap_img(im1, im2, a=0.5, b=0.5):
#     return a * im1 + b * im2

# def tensor2img(tensor):
#     return tensor.permute(1,2,0).cpu().detach().numpy()

# def tensor2uint8(tensor):
#     return img2uint8(tensor2img(tensor))

# def img2uint8(array):
#     return (array * 255).astype(np.uint8)

# def _cumulative_sum_threshold(values, percentile):
#     sorted_vals = np.sort(values.flatten())
#     cum_sums = np.cumsum(sorted_vals)
#     threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
#     return sorted_vals[threshold_id]

# def _normalize_scale(attr, scale_factor):
#     attr_norm = attr / scale_factor
#     return np.clip(attr_norm, 0, 1)

# def normalize_attribution(attr):
#     attr = attr[0].permute(1,2,0).detach().cpu().numpy()
#     attr_combined = np.abs(np.sum(attr, axis=2))
#     threshold = _cumulative_sum_threshold(attr_combined, 100 - 2)
#     norm_attr = _normalize_scale(attr_combined, threshold)    
#     attr_map = cv2.applyColorMap(img2uint8(norm_attr), cv2.COLORMAP_HOT)
#     attr_map = np.float32(attr_map) / 255
#     attr_map = attr_map[..., ::-1]
#     return attr_map

# def find_high_activation_crop(activation_map, percentile=95):
#     threshold = np.percentile(activation_map, percentile)
#     mask = np.ones(activation_map.shape)
#     mask[activation_map < threshold] = 0
#     lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
#     for i in range(mask.shape[0]):
#         if np.amax(mask[i]) > 0.9:
#             lower_y = i
#             break
#     for i in reversed(range(mask.shape[0])):
#         if np.amax(mask[i]) > 0.9:
#             upper_y = i
#             break
#     for j in range(mask.shape[1]):
#         if np.amax(mask[:,j]) > 0.9:
#             lower_x = j
#             break
#     for j in reversed(range(mask.shape[1])):
#         if np.amax(mask[:,j]) > 0.9:
#             upper_x = j
#             break
#     return lower_y, upper_y+1, lower_x, upper_x+1

# def view(tensor, unormalizer=None, save_path=None):    
#     grid = make_grid(tensor, padding=5)
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     img = Image.fromarray(ndarr)
#     if save_path is not None:
#         img.save(save_path, format=None)
#     return img

# def view_gif(filename):
#     return HTML('<img style="margin-left: 0px" src="{}">'.format(filename))

# def view_group(files):
#     html_str = ''
#     for file in files:
#         html_str += '<img style="margin: 0px; display: inline; padding:5px" src="{}">'.format(file)
#     return HTML(html_str)

# def view_group_pretty(files):
#     html_str = ''
#     html_str += '<img style="margin: 0px; display: inline; padding:0px; width:150px" src="{}">'.format(files[0])
#     html_str += '<img style="margin: 0px; display: inline; padding:0px; width:150px" src="{}">'.format(files[1])
#     html_str += '<img style="margin: 0px; display: inline; padding:0px 10px 0px 20px; width:100px" src="{}">'.format(files[2])
#     html_str += '<img style="margin: 0px; display: inline; padding:0px 20px 0px 10px; width:100px" src="{}">'.format(files[3])
#     html_str += '<img style="margin: 0px; display: inline; padding:0px; width:150px" src="{}">'.format(files[4])
#     html_str += '<img style="margin: 0px; display: inline; padding:0px; width:150px" src="{}">'.format(files[5])
#     return HTML(html_str)