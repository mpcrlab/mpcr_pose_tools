import pims
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class Rescale(object):
    def __init__(self, scale=1/255.0):
        self.scale = scale
        
    def __call__(self, data):
        return data * self.scale
    
class ChannelsFirst(object):
    def __call__(self, data):
        return data.permute(3, 0, 1, 2)
    
class ChannelsLast(object):
    def __call__(self, data):
        return data.permute(1, 2, 3, 0)
    
class Resize(object):
    def __init__(self, size):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size
        
    def __call__(self, data):
        return F.interpolate(data, size=self.size, mode='bilinear')

class VideoFolder(Dataset):
    def __init__(self, videos_csv, chunk_size, step_size=None, stride=1, transform=None):
        self.videos_csv = open(videos_csv, "r").readlines()
        self.chunk_size = chunk_size
        self.step_size = step_size if step_size else chunk_size
        self.stride = stride
        self.transform = transform
        
        self.videos = []
        self.chunks = []
        self.classes = {}
        
        video_id = 0
        
        for line in self.videos_csv:
            video_path, class_name = line.strip("\n").split(",")
            
            if class_name not in self.classes.values():
                self.classes[len(self.classes.keys())] = class_name
               
            video = self.load_video(video_path)
            self.videos.append(video)
            
            for chunk_start in range(0, len(video), self.step_size * self.stride):
                chunk_info = (video_id, chunk_start, label)
                self.chunks.append(chunk_info)

            video_id += 1


    def __len__(self):
        return len(self.chunks)
    
    def load_video(self, path):
        return pims.Video(path)
                           
    def get_frames_from_video(self, video_id, chunk_start):
        video = self.videos[video_id]
        frames = video[chunk_start : min(len(video) - 1, chunk_start + (self.chunk_size * self.stride)) : self.stride]
        tensor = torch.tensor(np.array(frames), dtype=torch.float)
        
        if tensor.shape[0] != self.chunk_size:
            padded_tensor = torch.zeros((self.chunk_size, tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            padded_tensor[:tensor.shape[0],:,:,:] = tensor
            tensor = padded_tensor
            
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor


    def __getitem__(self, idx):
        (video_id, chunk_start, label) = self.chunks[idx]
        tensor = self.get_frames_from_video(video_id, chunk_start)
        return tensor, label
