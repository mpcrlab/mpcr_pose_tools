import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import random
import matplotlib.pyplot as plt

class CoordsDataset(Dataset):
    def __init__(self, data_dir, height=480, width=640, start_cutoff=10, end_cutoff=10, segment_length=20, stride=1, fps=25):
        super(CoordsDataset, self).__init__()
        self.data_dir = data_dir

        self.height, self.width = height, width
        self.fps = fps
        self.start_cutoff = start_cutoff * self.fps
        self.end_cutoff = end_cutoff * self.fps
        self.stride = round(stride * self.fps)
        self.segment_length = segment_length * self.fps

        self.segments, self.segment_labels, self.labels_dict = self.get_data_and_labels(self.data_dir)


    def get_data_and_labels(self, data_dir):
        segments, segment_labels = [], []

        labels = sorted(os.listdir(data_dir))
        labels_dict = {label: i for i, label in enumerate(labels)}

        for label in labels:
            label_path = os.path.join(data_dir, label)
            sequences = sorted(os.listdir(label_path))
            for sequence in sequences:
                sequence_path = os.path.join(label_path, sequence)
                timesteps = sorted(os.listdir(sequence_path))

                for i in range(self.start_cutoff, len(timesteps) - self.end_cutoff - self.segment_length, self.segment_length):
                    segment = []
                    for j in range(i, i+self.segment_length, self.stride):
                        timestep_path = os.path.join(sequence_path, timesteps[j])
                        segment.append(timestep_path)
                    segments.append(segment)
                    segment_labels.append(labels_dict[label])

        return segments, segment_labels, labels_dict

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment, label = self.segments[idx], self.segment_labels[idx]
        data = []
        for coord_path in segment:
            data.append(self.get_coords(coord_path))
        data = np.stack(data)

        return torch.tensor(data), label

    def get_coords(self, coord_path):
        coords = np.load(coord_path)['pose']

        num_people = coords.shape[0]
        num_keypoints = coords.shape[1]
        num_features = coords.shape[2]

        if num_people == 0:
            return np.zeros(num_keypoints*num_features, dtype=np.float32)

        coords[0,:,0] /= float(self.width)
        coords[0,:,1] /= float(self.height)
        return coords[0].reshape(num_keypoints * num_features).astype(np.float32)

def get_train_val_test(dataset, batch_size=32, train=0.8, val=0.1, test=0.1):
    assert train + val + test == 1

    train_length = int(len(dataset) * train)
    val_length = int(len(dataset) * val)
    test_length = len(dataset) - train_length - val_length

    trainset, valset, testset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    valloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
    testloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(0,1)
        output, hidden = self.rnn(x)
        return self.fc(output[-1])

def binary_accuracy(preds, y):
    _, max_idxs = preds.max(1)
    correct = (max_idxs == y).float()
    return correct.sum()/len(correct)

def train_epoch(model, loader, optimizer):
    loss_epoch, acc_epoch = 0, 0
    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        acc = binary_accuracy(output, labels)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        acc_epoch += acc.item()

    return loss_epoch / len(loader), acc_epoch / len(loader)

def test_nn(data_dir):
    dataset = CoordsDataset(data_dir)
    trainloader, valloader, testloader = get_train_val_test(dataset)

    model = RNN(75, 100, 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(30):
        loss, acc = train_epoch(model, trainloader, optimizer)
        print(F"Epoch: {epoch+1} Loss: {loss} \tAcc: {acc}")

def test_movement(data_dir):
    dataset = CoordsDataset(data_dir)

    std_devs = []
    labels = []
    for segment, label in dataset:
        segment = segment.view(-1, 25, 3)
        mean = segment.mean(dim=[0,1]).numpy()
        std_dev = segment.std(dim=[0,1]).numpy()
        std_devs.append(std_dev)
        labels.append(label)
    std_devs = np.stack(std_devs)
    labels = np.array(labels)
    plt.scatter(std_devs[labels==0,0], std_devs[labels==0,1], label='disengaged')
    plt.scatter(std_devs[labels==1,0], std_devs[labels==1,1], label='engaged')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_nn(data_dir = "/home/vmlubuntu/Documents/labeled pose study 3/Coords/")
