import numpy as np
import torchvision
import torch
from sklearn.model_selection import train_test_split
from video_loader import MinMaxScaler
from video_loader import VideoLoader
from video_loader import VideoClipLoader

def load_balls_data(batch_size):
    
    with open('ball_videos.npy', 'rb') as fin:
        ball_videos = torch.tensor(np.load(fin)).float()

    batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(ball_videos,
                                               batch_size=batch_size,
                                               shuffle=True)

    return train_loader

def load_bamc_data(batch_size):   
    video_path = "/home/cm3786@drexel.edu/bamc_data/"
    
    scale = 0.2
    
    base_width = 1920
    base_height = 1080
    
    cropped_width = round(140/320 * base_width)
    cropped_height = round(140/180 * base_height)
    
    width = round(cropped_width * scale)
    height = round(cropped_height * scale)
    
    transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                 torchvision.transforms.CenterCrop(size=(cropped_height, cropped_width)),
                                                 torchvision.transforms.Resize(size=(width, height)), 
                                                 MinMaxScaler(0, 255)])
    dataset = VideoLoader(video_path, transform=transforms, num_frames=60)
    
    targets = dataset.get_labels()
    
    train_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    # shuffle=True,
                                                    sampler=test_sampler)

    return train_loader, test_loader

def load_covid_data(batch_size, clip_length_in_frames=10, frame_rate=20):   
    video_path = "/home/cm3786@drexel.edu/Projects/covid19_ultrasound/data/pocus_videos/convex"
    # video_path = "/home/cm3786@drexel.edu/Projects/covid19_ultrasound/data/pocus_videos/pneumonia-viral"
    
    scale = 0.5
    
    base_width = 1920
    base_height = 1080
    
    cropped_width = round(140/320 * base_width)
    cropped_height = round(140/180 * base_height)
    
    #width = round(cropped_width * scale)
    #height = round(cropped_height * scale)
    
    width = 128
    height = 128
    
    transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                 #torchvision.transforms.Resize(size=(base_width, base_height)),
                                                 #torchvision.transforms.CenterCrop(size=(cropped_height, cropped_width)),
                                                 torchvision.transforms.Resize(size=(width, height)), 
                                                 MinMaxScaler(0, 255)])
    dataset = VideoClipLoader(video_path, transform=transforms,
                              clip_length_in_frames=clip_length_in_frames,
                              frame_rate=frame_rate)
    
    targets = dataset.get_video_labels()
    train_vidx, test_vidx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    
    train_vidx = set(train_vidx)
    test_vidx = set(test_vidx)
    
    train_cidx = [i for i in range(len(dataset)) if dataset.video_idx[i] in train_vidx]
    test_cidx = [i for i in range(len(dataset)) if dataset.video_idx[i] in test_vidx]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_cidx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_cidx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    return train_loader, test_loader