import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import os
import math
import numpy as np
import random
import cv2

class CoordsDataset(Dataset):
    def __init__(self, data_dir, max_people=0, pad_max_people=True,
                 frames_per_segment=0, pad_frames=True, 
                 frame_stride=1, segment_stride=0, 
                 start_percent_skip=0.0, end_percent_skip=0.0,
                 keypoint_types=['pose', 'face', 'handl', 'handr'],
                 file_types=['.npz']):
        super(CoordsDataset, self).__init__()

        self.data_dir = data_dir
        self.max_people = max_people
        self.pad_max_people = pad_max_people
        
        self.frames_per_segment = frames_per_segment
        self.pad_frames = pad_frames
        
        if self.frames_per_segment <= 0:
            self.pad_frames = False
            
        self.frame_stride = max(frame_stride, 1)
        self.segment_stride = segment_stride
        
        if self.segment_stride <= 0:
            self.segment_stride = self.frames_per_segment
        
        self.start_percent_skip = start_percent_skip
        self.end_percent_skip = end_percent_skip
        
        self.keypoint_types = keypoint_types
        self.file_types = file_types

        self.segments, self.labels = [], []
        
        self.class_names = sorted(os.listdir(self.data_dir))
        for class_idx, class_folder in enumerate(self.class_names):
            class_path = os.path.join(self.data_dir, class_folder)
            
            for path, dirs, files in os.walk(class_path):
                if len(files) == 0:
                    continue
                
                relpath = os.path.relpath(path, self.data_dir)
                coords = list(filter(self._is_coords_file, files))
                coords.sort(key=self._numeric_sort_fn)
                
                num_frames = len(coords)
                start_frames_skip = int(self.start_percent_skip * num_frames)
                end_frames_skip = int(self.end_percent_skip * num_frames)
                
                frames_per_segment = self.frames_per_segment
                segment_stride = self.segment_stride
                if frames_per_segment <= 0:
                    frames_per_segment = num_frames - start_frames_skip - end_frames_skip
                    segment_stride = frames_per_segment
                
                start_idx = start_frames_skip
                end_idx = num_frames - end_frames_skip
                for start_frame in range(start_idx, end_idx, segment_stride):
                    segment_coords = []
                    
                    end_frame = min(start_frame + frames_per_segment, num_frames)
                    for i in range(start_frame, end_frame, self.frame_stride):
                        segment_coords.append(coords[i])
                
                    if len(segment_coords) > 0:
                        self.segments.append((relpath, segment_coords))
                        self.labels.append(class_idx)
                
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        sequence_path, files = self.segments[idx]
        
        coord_arrays = []
        for file in files:
            file_path = os.path.join(self.data_dir, sequence_path, file)
            coord_array = self.load_keypoints(file_path)
            coord_arrays.append(coord_array)

        coord_array = self.pad_keypoints(coord_arrays)
        
        return coord_array, self.labels[idx]
    
    def load_keypoints(self, file_path):
        coords = np.load(file_path)
        
        coord_arrays = []
        for key in self.keypoint_types:
            if key in coords:
                coord_arrays.append(coords[key])
                
        return np.concatenate(coord_arrays, axis=1)
    
    def pad_keypoints(self, keypoint_list):
        people_count = max([kp.shape[0] for kp in keypoint_list])
        if self.max_people > 0:
            people_count = min(people_count, self.max_people)
            if self.pad_max_people:
                people_count = self.max_people
            
        num_frames = len(keypoint_list)
        num_coords, num_feats = keypoint_list[0].shape[1:]
        
        if self.pad_frames and self.frames_per_segment > 0:
            num_frames = int(self.frames_per_segment / self.frame_stride)
        
        shape = (num_frames, people_count, num_coords, num_feats)
        padded = np.zeros(shape)
        
        for i, kp in enumerate(keypoint_list):
            people = min(kp.shape[0], people_count)
            padded[i,:people,:,:] = kp[:people,:,:]
        
        return padded
                    
    def _is_coords_file(self, file):
        fname, ext = os.path.splitext(file)
        return ext in self.file_types
    
    def _numeric_sort_fn(self, file):
        name, ext = os.path.splitext(file)
        if name.isnumeric():
            return int(name)
        return math.inf
    
def get_train_val_test_loaders(dataset, batch_size=32, train=0.8, val=0.1, test=0.1, num_workers=4):
    assert train + val + test == 1

    train_length = int(len(dataset) * train)
    val_length = int(len(dataset) * val)
    test_length = len(dataset) - train_length - val_length

    trainset, valset, testset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader

