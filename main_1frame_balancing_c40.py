import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

from model import dataloader
from model import create_dpnet

from utils import *
import push_1frame as push

def main():
    # =================== Arguments ===================
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       default=1429429,   type=int)
    parser.add_argument('--gpu',        default='0',       type=str)
    parser.add_argument('--record_dir', default='record/1frame_c40', type=str)
    parser.add_argument('--task',       default='Deepfakes', type=str)

    parser.add_argument('--num_workers',     default=8,    type=int)
    parser.add_argument('--num_epochs',      default=20,   type=int)
    parser.add_argument('--num_warm_epochs', default=1,    type=int)
    parser.add_argument('--batch_size',      default=64,   type=int)
    parser.add_argument('--checkpoint',      default=None, type=str)
    args = parser.parse_args()
    print(args)

    # =================== Settings ===================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(args.record_dir, args.task, 'seed_{}'.format(args.seed))
    makedir(log_dir)

    # =================== Dataloaders ===================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    data_dir = '/meladyfs/newyork/loctrinh/DATASETS/'
    frame_count = {'FF++': pd.read_csv(os.path.join(data_dir, 'FF++', 'video_stat.csv'), index_col=0)}
    train_df = pd.read_csv(os.path.join(data_dir, 'FF++/splits/{}_trainlist_01.csv'.format(args.task)))
    val_df = pd.read_csv(os.path.join(data_dir, 'FF++/splits/{}_vallist_01.csv'.format(args.task)))
    test_df = pd.read_csv(os.path.join(data_dir, 'FF++/splits/{}_testlist_01.csv'.format(args.task)))

    data_loader = dataloader.SingleC40ImageLoader(args.batch_size, args.num_workers, data_dir, frame_count,
                                               train_df, val_df, test_df, train_transform, test_transform)
    train_loader, val_loader, test_loader, push_loader = data_loader.run()
    
    # =================== Training =================== 
    detector = DPNet(device=device,
                        log_dir=log_dir,
                        args=args,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        push_loader=push_loader)
    detector.train()
    detector.test()
    
class DPNet():
    def __init__(self, device, log_dir, args, train_loader, val_loader, test_loader, push_loader):
        self.device = device
        self.log_dir = log_dir
        self.args = args
        
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.push_loader = push_loader
        
        self.best_val_loss = float('inf')
        self.counter = 0
        self.patience = 40
        self.early_stop = False
        
        self.coefs = {
            'crs_ent': 1.0,
            'clst': 0.2,
            'sep': -0.2,
            'div': 0,
            'l1': 1e-3,
        }    
        
        self.build_model()
        
    def build_model(self):        
        self.model = create_dpnet(in_channel=3)
        self.model = nn.DataParallel(self.model)

        balancing_weight = torch.tensor([2, 0.5]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(balancing_weight)
        self.warm_optimizer = torch.optim.Adam([{'params': self.model.module.add_on_layers.parameters(), 'lr': 1e-3},
                                                {'params': self.model.module.prototype_vectors, 'lr': 1e-3}])
        self.full_optimizer = torch.optim.Adam([{'params': self.model.module.features.parameters(), 'lr': 2e-4},
                                                {'params': self.model.module.add_on_layers.parameters(), 'lr': 1e-3},
                                                {'params': self.model.module.prototype_vectors, 'lr': 1e-3}])
        
        if self.args.checkpoint:
            cp = torch.load(self.args.checkpoint)
            self.epoch = cp['epoch']
            self.model.load_state_dict(cp['state_dict'])

        self.model = self.model.to(self.device)
        
    def train(self):
        for self.epoch in range(self.args.num_epochs):  
            if self.epoch < self.args.num_warm_epochs:
                epoch_optimizer = self.warm_optimizer
                self.train_1epoch('warm', self.warm_optimizer)
            else:
                epoch_optimizer = self.full_optimizer
                self.train_1epoch('train', self.full_optimizer)
                self.push_1epoch()
            
            if self.early_stop:
                break
                
        self.model.load_state_dict(torch.load(os.path.join(self.log_dir, 'best_val_checkpoint.pth'))['state_dict'])      
        
    def test(self):
        self.validate_1epoch(test_mode=True)
        
    def push_1epoch(self):
        print('==> Epoch:[{0}/{1}][pushing stage]'.format(self.epoch, self.args.num_epochs))
        push.push_prototypes(
            self.device,
            self.push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=self.model, # pytorch network with prototype_vectors
            class_specific=True,
            preprocess_input_function=None, # normalize if needed
            prototype_layer_stride=1,
            epoch_number=self.epoch, # if not provided, prototypes saved previously will be overwritten
            root_dir_for_saving_prototypes    =os.path.join(self.log_dir, 'prototypes'),
            prototype_img_filename_prefix     ='prototype_',
            prototype_self_act_filename_prefix='prototype_act_',
            proto_bound_boxes_filename_prefix ='prototype_bb_',
            save_prototype_class_identity=True)
            
    def train_1epoch(self, mode, optimizer):
        # Setting trainable parameters
        if mode == 'warm':
            print('|--> Epoch:[{0}/{1}][warm stage]'.format(self.epoch+1, self.args.num_epochs))
            for param in self.model.module.features.parameters():
                param.requires_grad = False
            for param in self.model.module.add_on_layers.parameters():
                param.requires_grad = True
            self.model.module.prototype_vectors.requires_grad = True
        elif mode == 'train':
            print('|--> Epoch:[{0}/{1}][train stage]'.format(self.epoch+1, self.args.num_epochs))
            for param in self.model.parameters():
                param.requires_grad = True
        
        losses, top1 = AverageMeter(), AverageMeter()
        ce, clus, sep, div, el1 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        # Mini-batch training
        start = time.time()
        progress = tqdm(self.train_loader)
        for iteration, (inputs, labels) in enumerate(progress):
             # Train mode
            self.model.train()  
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Compute output
            batch_size = inputs.shape[0]
            outputs, min_distances = self.model(inputs)
            max_dist = (self.model.module.prototype_shape[1] 
                        * self.model.module.prototype_shape[2] 
                        * self.model.module.prototype_shape[3])            
            
            # ===== LOSS =====
            # calculate ce cost
            cross_entropy = self.criterion(outputs, labels)

            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.model.module.prototype_class_identity[:,labels]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances) / max_dist
            
            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes) / max_dist       
            
            l1_mask = 1 - torch.t(self.model.module.prototype_class_identity).cuda()
            l1 = (self.model.module.last_layer.weight * l1_mask).norm(p=1)
            
            loss = self.coefs['crs_ent'] * cross_entropy + \
                    self.coefs['clst'] * cluster_cost + \
                    self.coefs['sep'] * separation_cost + \
                    self.coefs['l1'] * l1
              
            # Measure accuracy and record loss
            acc = accuracy(outputs.data, labels, topk=(1, ))                 
            losses.update(loss.item(), batch_size)
            top1.update(acc.item(), batch_size)
            
            ce.update(cross_entropy.item(), batch_size)
            clus.update(cluster_cost.item(), batch_size)
            sep.update(separation_cost.item(), batch_size)
            el1.update(l1.item(), batch_size)

            # Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
             # Validating  
            if iteration > 0 and iteration % 4000 == 0:
                print({'Epoch': [self.epoch],
                 'Time':  [round(time.time()-start,3)],
                 'Loss':  [round(losses.avg,5)],
                 'Acc':   [round(top1.avg,4)],
                 'ce':  [round(ce.avg,5)],
                 'clus':[round(clus.avg,5)],
                 'sep': [round(sep.avg,5)],
                 'el1': [round(el1.avg,5)]})
                print(f'\nValidation check at step {iteration}')
                prec1, auc, val_loss = self.validate_1epoch()
                if val_loss < self.best_val_loss - 1e-2:
                    self.best_val_loss = val_loss
                    if mode == 'train': self.push_1epoch()
                    print(f'Saving predictions with best loss: {val_loss} and AUC: {auc}')
                    save_predictions(self.dic_video_level_preds, os.path.join(self.log_dir, 'best_val_predictions.pickle'))
                    save_checkpoint({
                        'epoch': self.epoch,
                        'loss': self.best_val_loss,
                        'state_dict': self.model.state_dict(),
                        'optimizer_dict' : optimizer.state_dict()
                    }, os.path.join(self.log_dir, 'best_val_checkpoint.pth'))
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        print(f'===== EarlyStopped =====')
                        self.early_stop = True
                        break
                        
        info = {'Epoch': [self.epoch],
                 'Time':  [round(time.time()-start,3)],
                 'Loss':  [round(losses.avg,5)],
                 'Acc':   [round(top1.avg,4)],
                 'ce':  [round(ce.avg,5)],
                 'clus':[round(clus.avg,5)],
                 'sep': [round(sep.avg,5)],
                 'el1': [round(el1.avg,5)]}
        record_info(info, os.path.join(self.log_dir, 'train.csv'))
    
    def validate_1epoch(self, test_mode=False):
        if test_mode:
            print('|--> [[testing stage]]')
        else:
            print('|--> Epoch:[{0}/{1}][validation stage]'.format(self.epoch+1, self.args.num_epochs))

        losses, top1 = AverageMeter(), AverageMeter()

        # Evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        
        start = time.time()
        with torch.no_grad():
            progress = tqdm(self.test_loader) if test_mode else tqdm(self.val_loader)
            for _, (video_names, inputs, labels) in enumerate(progress):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Compute output
                batch_size = inputs.shape[0]             
                outputs, min_distances = self.model(inputs)

                # Accumulate video level prediction
                preds = outputs.data.cpu().numpy()
                for i in range(batch_size):
                    video_name = video_names[i]
                    if video_name not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[video_name] = preds[i,:]
                    else:
                        self.dic_video_level_preds[video_name] += preds[i,:]
        
        # Calculate video level statistics
        video_top1, video_auc, video_loss = self.frame_2_video_level_accuracy()

        info = {'Epoch': [self.epoch],
                'Time':  [round(time.time()-start,3)],
                'Loss':  [round(video_loss,5)],
                'Acc':   [round(video_top1,4)],
                'AUC':   [round(video_auc,4)]}
        if test_mode:
            print(info)
        else:
            record_info(info, os.path.join(self.log_dir, 'test.csv'))
        return video_top1, video_auc, video_loss
                             
    def frame_2_video_level_accuracy(self):
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),2))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        
        for i, name in enumerate(sorted(self.dic_video_level_preds.keys())):
            preds = self.dic_video_level_preds[name]
            label = 1.0 if 'FAKE' in name else 0.0
                
            video_level_preds[i,:] = preds / 100
            video_level_labels[i] = label
            if np.argmax(preds) == (label):
                correct += 1

        video_level_labels = torch.from_numpy(video_level_labels).long().to(self.device)
        video_level_preds = torch.from_numpy(video_level_preds).float().to(self.device)
            
        top1 = accuracy(video_level_preds, video_level_labels, topk=(1,))
        loss = self.criterion(video_level_preds, video_level_labels)
                                 
        logits = nn.functional.softmax(video_level_preds, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(video_level_labels.cpu(), logits)
        
        return top1.item(), auc, loss.item()  

                                 
if __name__=='__main__':
    main()
                                 
