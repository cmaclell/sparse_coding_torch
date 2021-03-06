from os import listdir
from os.path import isfile
from os.path import join
from os.path import isdir
from os.path import abspath
from torchvision.datasets.video_utils import VideoClips

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
import torchvision as tv
from torch import nn

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self, min_val=0, max_val=254):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

class VideoGrayScaler(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        
    def forward(self, video):
        # shape = channels, time, width, height
        video = self.grayscale(video.swapaxes(-4, -3).swapaxes(-2, -1))
        video = video.swapaxes(-4, -3).swapaxes(-2, -1)
        # print(video.shape)
        return video
    
class VideoLoader(Dataset):
    
    def __init__(self, video_path, transform=None, num_frames=None):
        self.num_frames = num_frames
        self.transform = transform
        
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
            
        self.cache = {}
        
    def get_labels(self):
        return [self.videos[i][0] for i in range(len(self.videos))]
    
    def __getitem__(self, index):
        #print('index: {}'.format(index))
        
        if index in self.cache:
            return self.cache[index]
        
        label = self.videos[index][0]
        filename = self.videos[index][2]
        video, _, info = read_video(self.videos[index][1])
        # print(info)
        video = torch.swapaxes(video, 1, 3)
        
        # print('length', len(video))
        if self.num_frames:
            video = video[-self.num_frames:]
            
            if len(video) < self.num_frames:
                padding = torch.zeros(self.num_frames - len(video), video.shape[1], video.shape[2], video.shape[3])
                video = torch.cat((video, padding))
            
        video = video.swapaxes(0, 1).swapaxes(2, 3)
        
        if self.transform:
            video = self.transform(video)
        
        self.cache[index] = (label, video, filename)
            
        return label, video, filename
        
    def __len__(self):
        return len(self.videos)
    
class VideoClipLoader(Dataset):
    
    def __init__(self, video_path, clip_length_in_frames=20, frame_rate=20, frames_between_clips=None, transform=None):
        self.transform = transform
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
            
        #for v in self.videos:
        #    video, _, info = read_video(v[1])
        #    print(video.shape)
        #    print(info)
            
        if not frames_between_clips:
            frames_between_clips = clip_length_in_frames
            
        vc = VideoClips([path for _, path, _ in self.videos],
                        clip_length_in_frames=clip_length_in_frames,
                        frame_rate=frame_rate,
                       frames_between_clips=frames_between_clips)
        self.clips = []
                   
        self.video_idx = []
        for i in tqdm(range(vc.num_clips())):
            try:
                clip, _, _, vid_idx = vc.get_clip(i)
                clip = clip.swapaxes(1, 3).swapaxes(2, 3)
                clip = clip.swapaxes(0, 1)
                if self.transform:
                    clip = self.transform(clip)
                self.clips.append((self.videos[vid_idx][0], clip, self.videos[vid_idx][2]))
                self.video_idx.append(vid_idx)
            except Exception:
                pass
        
    def get_video_labels(self):
        return [self.videos[i][0] for i in range(len(self.videos))]
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def __getitem__(self, index):
        return self.clips[index]
        
    def __len__(self):
        return len(self.clips)

if __name__ == "__main__":
    video_path = "/shared_data/bamc_data/"
    
    transforms = tv.transforms.Compose([VideoGrayScaler()])

    # dataset = VideoLoader(video_path, transform=transforms, num_frames=60)
    dataset = VideoClipLoader(video_path, transform=transforms, clip_length_in_frames=20)
    #for data in dataset:
    #    print(data[0], data[1].shape)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True)
    
    for data in loader:
        print(data[0], data[1].shape, data[2])
        #print(data)