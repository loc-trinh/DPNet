import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StackNImageDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.keys)
    
    def get_stack(self, video_name, frame_index):
        img_dir = video_name.replace('video_data', 'frames')[:-4]
        data = torch.FloatTensor(18, 256, 256)
        
        img_path = os.path.join(self.root_dir, img_dir, 'frame' + str(frame_index).zfill(5) + '.png')
        data[:3,:,:] = self.transform((Image.open(img_path)))
        for j in range(15):
            idx = frame_index + j * 1
            img_path = os.path.join(self.root_dir, img_dir, 'flow' + str(idx).zfill(5) + '.png')
            data[j+3,:,:] = self.to_tensor((Image.open(img_path)))
        return data

    def __getitem__(self, idx):
        label = int(self.values[idx])
        if self.mode == 'train':
            video_name, frame_index = self.keys[idx].split(' ')         
            data = self.get_stack(video_name, int(frame_index))
            sample = (data, label)
        elif self.mode == 'test':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, data, label)
        elif self.mode == 'push':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, frame_index, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class StackNImageLoader():
    def __init__(self, batch_size, num_workers, data_dir, frame_count,
                 train_df, val_df, test_df, train_transform, test_transform):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.frame_count = frame_count
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = train_transform
        self.test_transform = test_transform
        
    def sample_video_to_dic(self, df, sample_size):
        video_dic = {}
        for row in df.itertuples():
            frame_df = self.frame_count[row.video.split('/')[0]]
            num_frame = int(frame_df.loc[frame_df.video == row.video].num_frames) - 15
            num_sample = num_frame - 15 if num_frame - 15 < sample_size else sample_size
            if num_sample <= 0:
                continue
            
            interval = int(num_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(row.video, str(frame))
                video_dic[key] = row.label
        return video_dic

    def get_train_loader(self, num_sample):
        training_dic = self.sample_video_to_dic(self.train_df, num_sample)
        training_set = StackNImageDataset(dic=training_dic, root_dir=self.data_dir, mode='train', transform=self.train_transform)
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self, num_sample):
        validation_dic = self.sample_video_to_dic(self.val_df, num_sample)
        validation_set = StackNImageDataset(dic=validation_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def get_test_loader(self, num_sample):
        testing_dic = self.sample_video_to_dic(self.test_df, num_sample)
        testing_set = StackNImageDataset(dic=testing_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Testing data:', len(testing_set), 'frames')

        test_loader = DataLoader(
            dataset=testing_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader
    
    def get_push_loader(self, num_sample):
        pushing_dic = self.sample_video_to_dic(self.train_df, num_sample)
        pusing_set = StackNImageDataset(dic=pushing_dic, root_dir=self.data_dir, mode='push', transform=self.train_transform)
        print('==> Pushing data:', len(pusing_set),'frames')

        push_loader = DataLoader(
            dataset=pusing_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return push_loader
    
    def run(self):
        train_loader = self.get_train_loader(270)
        val_loader   = self.get_val_loader(100)
        test_loader  = self.get_test_loader(100)
        push_loader  = self.get_push_loader(100)
        return train_loader, val_loader, test_loader, push_loader



class StackC40ImageDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.keys)
    
    def get_stack(self, video_name, frame_index):
        img_dir = video_name.replace('video_data', 'frames')[:-4].replace('c23', 'c40')
        data = torch.FloatTensor(12, 256, 256)
        
        img_path = os.path.join(self.root_dir, img_dir, 'frame' + str(frame_index).zfill(5) + '.png')
        data[:3,:,:] = self.transform((Image.open(img_path)))
        for j in range(9):
            idx = frame_index + j * 1
            img_path = os.path.join(self.root_dir, img_dir, 'flow' + str(idx).zfill(5) + '.png')
            data[j+3,:,:] = self.to_tensor((Image.open(img_path)))
        return data

    def __getitem__(self, idx):
        label = int(self.values[idx])
        if self.mode == 'train':
            video_name, frame_index = self.keys[idx].split(' ')         
            data = self.get_stack(video_name, int(frame_index))
            sample = (data, label)
        elif self.mode == 'test':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, data, label)
        elif self.mode == 'push':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, frame_index, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class StackC40ImageLoader():
    def __init__(self, batch_size, num_workers, data_dir, frame_count,
                 train_df, val_df, test_df, train_transform, test_transform):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.frame_count = frame_count
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = train_transform
        self.test_transform = test_transform
        
    def sample_video_to_dic(self, df, sample_size):
        video_dic = {}
        for row in df.itertuples():
            frame_df = self.frame_count[row.video.split('/')[0]]
            num_frame = int(frame_df.loc[frame_df.video == row.video].num_frames) - 10
            num_sample = num_frame - 10 if num_frame - 10 < sample_size else sample_size
            if num_sample <= 0:
                continue
            
            interval = int(num_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(row.video, str(frame))
                video_dic[key] = row.label
        return video_dic

    def get_train_loader(self, num_sample):
        training_dic = self.sample_video_to_dic(self.train_df, num_sample)
        training_set = StackC40ImageDataset(dic=training_dic, root_dir=self.data_dir, mode='train', transform=self.train_transform)
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self, num_sample):
        validation_dic = self.sample_video_to_dic(self.val_df, num_sample)
        validation_set = StackC40ImageDataset(dic=validation_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def get_test_loader(self, num_sample):
        testing_dic = self.sample_video_to_dic(self.test_df, num_sample)
        testing_set = StackImageDataset(dic=testing_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Testing data:', len(testing_set), 'frames')

        test_loader = DataLoader(
            dataset=testing_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader
    
    def get_push_loader(self, num_sample):
        pushing_dic = self.sample_video_to_dic(self.train_df, num_sample)
        pusing_set = StackC40ImageDataset(dic=pushing_dic, root_dir=self.data_dir, mode='push', transform=self.train_transform)
        print('==> Pushing data:', len(pusing_set),'frames')

        push_loader = DataLoader(
            dataset=pusing_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return push_loader
    
    def run(self):
        train_loader = self.get_train_loader(270)
        val_loader   = self.get_val_loader(100)
        test_loader  = self.get_test_loader(100)
        push_loader  = self.get_push_loader(100)
        return train_loader, val_loader, test_loader, push_loader



class StackImageDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.keys)
    
    def get_stack(self, video_name, frame_index):
        img_dir = video_name.replace('video_data', 'frames')[:-4]
        data = torch.FloatTensor(12, 256, 256)
        
        img_path = os.path.join(self.root_dir, img_dir, 'frame' + str(frame_index).zfill(5) + '.png')
        data[:3,:,:] = self.transform((Image.open(img_path)))
        for j in range(9):
            idx = frame_index + j * 1
            img_path = os.path.join(self.root_dir, img_dir, 'flow' + str(idx).zfill(5) + '.png')
            data[j+3,:,:] = self.to_tensor((Image.open(img_path)))
        return data

    def __getitem__(self, idx):
        label = int(self.values[idx])
        if self.mode == 'train':
            video_name, frame_index = self.keys[idx].split(' ')         
            data = self.get_stack(video_name, int(frame_index))
            sample = (data, label)
        elif self.mode == 'test':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, data, label)
        elif self.mode == 'push':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_stack(video_name, int(frame_index))
            sample = (video_name, frame_index, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class StackImageLoader():
    def __init__(self, batch_size, num_workers, data_dir, frame_count,
                 train_df, val_df, test_df, train_transform, test_transform):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.frame_count = frame_count
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = train_transform
        self.test_transform = test_transform
        
    def sample_video_to_dic(self, df, sample_size):
        video_dic = {}
        for row in df.itertuples():
            frame_df = self.frame_count[row.video.split('/')[0]]
            num_frame = int(frame_df.loc[frame_df.video == row.video].num_frames) - 10
            num_sample = num_frame - 10 if num_frame - 10 < sample_size else sample_size
            if num_sample <= 0:
                continue
            
            interval = int(num_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(row.video, str(frame))
                video_dic[key] = row.label
        return video_dic

    def get_train_loader(self, num_sample):
        training_dic = self.sample_video_to_dic(self.train_df, num_sample)
        training_set = StackImageDataset(dic=training_dic, root_dir=self.data_dir, mode='train', transform=self.train_transform)
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self, num_sample):
        validation_dic = self.sample_video_to_dic(self.val_df, num_sample)
        validation_set = StackImageDataset(dic=validation_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def get_test_loader(self, num_sample):
        testing_dic = self.sample_video_to_dic(self.test_df, num_sample)
        testing_set = StackImageDataset(dic=testing_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Testing data:', len(testing_set), 'frames')

        test_loader = DataLoader(
            dataset=testing_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader
    
    def get_push_loader(self, num_sample):
        pushing_dic = self.sample_video_to_dic(self.train_df, num_sample)
        pusing_set = StackImageDataset(dic=pushing_dic, root_dir=self.data_dir, mode='push', transform=self.train_transform)
        print('==> Pushing data:', len(pusing_set),'frames')

        push_loader = DataLoader(
            dataset=pusing_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return push_loader
    
    def run(self):
        train_loader = self.get_train_loader(270)
        val_loader   = self.get_val_loader(100)
        test_loader  = self.get_test_loader(100)
        push_loader  = self.get_push_loader(100)
        return train_loader, val_loader, test_loader, push_loader


class SingleC40ImageDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)
   
    def get_frame(self, video_name, frame_index):
        img_dir = video_name.replace('video_data', 'frames')[:-4].replace('c23', 'c40')
        img_path = os.path.join(self.root_dir, img_dir, 'frame' + str(frame_index).zfill(5) + '.png')
        img = (Image.open(img_path))
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        label = int(self.values[idx])
        if self.mode == 'train':
            video_name, frame_index = self.keys[idx].split(' ')         
            data = self.get_frame(video_name, int(frame_index))
            sample = (data, label)
        elif self.mode == 'test':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_frame(video_name, int(frame_index))
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class SingleC40ImageLoader():
    def __init__(self, batch_size, num_workers, data_dir, frame_count,
                 train_df, val_df, test_df, train_transform, test_transform):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.frame_count = frame_count
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = train_transform
        self.test_transform = test_transform
        
    def sample_video_to_dic(self, df, sample_size):
        video_dic = {}
        for row in df.itertuples():
            frame_df = self.frame_count[row.video.split('/')[0]]
            num_frame = int(frame_df.loc[frame_df.video == row.video].num_frames)
            num_sample = num_frame if num_frame < sample_size else sample_size
            
            interval = int(num_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(row.video, str(frame))
                video_dic[key] = row.label
        return video_dic

    def get_train_loader(self, num_sample):
        training_dic = self.sample_video_to_dic(self.train_df, num_sample)
        training_set = SingleC40ImageDataset(dic=training_dic, root_dir=self.data_dir, mode='train', transform=self.train_transform)
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self, num_sample):
        validation_dic = self.sample_video_to_dic(self.val_df, num_sample)
        validation_set = SingleC40ImageDataset(dic=validation_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def get_test_loader(self, num_sample):
        testing_dic = self.sample_video_to_dic(self.test_df, num_sample)
        testing_set = SingleC40ImageDataset(dic=testing_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Testing data:', len(testing_set), 'frames')

        test_loader = DataLoader(
            dataset=testing_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader
    
    def get_push_loader(self, num_sample):
        pushing_dic = self.sample_video_to_dic(self.train_df, num_sample)
        pusing_set = SingleC40ImageDataset(dic=pushing_dic, root_dir=self.data_dir, mode='test', transform=self.train_transform)
        print('==> Pushing data:', len(pusing_set),'frames')

        push_loader = DataLoader(
            dataset=pusing_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return push_loader
    
    def run(self):
        train_loader = self.get_train_loader(270)
        val_loader = self.get_val_loader(100)
        test_loader = self.get_test_loader(100)
        push_loader = self.get_push_loader(100)
        return train_loader, val_loader, test_loader, push_loader
    

class SingleImageDataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)
   
    def get_frame(self, video_name, frame_index):
        img_dir = video_name.replace('video_data', 'frames')[:-4]
        img_path = os.path.join(self.root_dir, img_dir, 'frame' + str(frame_index).zfill(5) + '.png')
        img = (Image.open(img_path))
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        label = int(self.values[idx])
        if self.mode == 'train':
            video_name, frame_index = self.keys[idx].split(' ')         
            data = self.get_frame(video_name, int(frame_index))
            sample = (data, label)
        elif self.mode == 'test':
            video_name, frame_index = self.keys[idx].split(' ')
            data = self.get_frame(video_name, int(frame_index))
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class SingleImageLoader():
    def __init__(self, batch_size, num_workers, data_dir, frame_count,
                 train_df, val_df, test_df, train_transform, test_transform):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.frame_count = frame_count
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = train_transform
        self.test_transform = test_transform
        
    def sample_video_to_dic(self, df, sample_size):
        video_dic = {}
        for row in df.itertuples():
            frame_df = self.frame_count[row.video.split('/')[0]]
            num_frame = int(frame_df.loc[frame_df.video == row.video].num_frames)
            num_sample = num_frame if num_frame < sample_size else sample_size
            
            interval = int(num_frame/num_sample)
            for i in range(num_sample):
                frame = i*interval
                key = '{} {}'.format(row.video, str(frame))
                video_dic[key] = row.label
        return video_dic

    def get_train_loader(self, num_sample):
        training_dic = self.sample_video_to_dic(self.train_df, num_sample)
        training_set = SingleImageDataset(dic=training_dic, root_dir=self.data_dir, mode='train', transform=self.train_transform)
        print('==> Training data:', len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def get_val_loader(self, num_sample):
        validation_dic = self.sample_video_to_dic(self.val_df, num_sample)
        validation_set = SingleImageDataset(dic=validation_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Validation data:', len(validation_set), 'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
    
    def get_test_loader(self, num_sample):
        testing_dic = self.sample_video_to_dic(self.test_df, num_sample)
        testing_set = SingleImageDataset(dic=testing_dic, root_dir=self.data_dir, mode='test', transform=self.test_transform)
        print('==> Testing data:', len(testing_set), 'frames')

        test_loader = DataLoader(
            dataset=testing_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader
    
    def get_push_loader(self, num_sample):
        pushing_dic = self.sample_video_to_dic(self.train_df, num_sample)
        pusing_set = SingleImageDataset(dic=pushing_dic, root_dir=self.data_dir, mode='test', transform=self.train_transform)
        print('==> Pushing data:', len(pusing_set),'frames')

        push_loader = DataLoader(
            dataset=pusing_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return push_loader
    
    def run(self):
        train_loader = self.get_train_loader(270)
        val_loader = self.get_val_loader(100)
        test_loader = self.get_test_loader(100)
        push_loader = self.get_push_loader(100)
        return train_loader, val_loader, test_loader, push_loader