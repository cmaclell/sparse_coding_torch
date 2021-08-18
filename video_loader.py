from os import listdir
from os.path import isfile, join, isdir, abspath

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
import torchvision as tv

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)


class VideoLoader(Dataset):
    
    def __init__(self, video_path, transform=None, num_frames=None):
        self.num_frames = num_frames
        self.transform = transform
        
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f))) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
        self.cache = {}
    
    def __getitem__(self, index):
        #print('index: {}'.format(index))
        
        if index in self.cache:
            return self.cache[index]
        
        label = self.videos[index][0]
        video, _, info = read_video(self.videos[index][1])
        video = torch.swapaxes(video, 1, 3)
        
        if self.num_frames:
            video = video[:self.num_frames]
        
        if self.transform:
            video = self.transform(video)
            
        video = video.swapaxes(0, 1).swapaxes(2, 3)
        
        self.cache[index] = (label, video)
            
        return label, video
        
    def __len__(self):
        return len(self.videos)

if __name__ == "__main__":
    video_path = "/home/cm3786@drexel.edu/bamc_data/"
    
    transforms = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                        tv.transforms.Resize(size=(320, 180))])

    dataset = VideoLoader(video_path, transform=transforms, num_frames=60)
    #for data in dataset:
    #    print(data[0], data[1].shape)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True)
    
    for data in loader:
        print(data[0], data[1].shape)
        #print(data)