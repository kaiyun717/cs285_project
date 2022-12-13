from os import path
import numpy as np
from torch.utils.data import Dataset
import torch

torch.set_default_dtype(torch.float32)

class OfflineDataset(Dataset):
    def __init__(self, dir, image_size, stack, channels):
        self.dir = dir

        dataset_dict = np.load(path.join(dir, 'dataframe.npz'))

        self.observations = torch.tensor(dataset_dict['observations'])
        self.rewards = torch.tensor(dataset_dict['rewards'])
        self.actions = torch.tensor(dataset_dict['actions'])
        self.next_observations = torch.tensor(dataset_dict['next_observations'])
        self.dones = torch.tensor(dataset_dict['dones'])

        if len(self.observations.shape) == 5:
            assert self.observations.shape[1:] == (stack, image_size, image_size, channels)
            assert self.next_observations.shape[1:] == (stack, image_size, image_size, channels)

            # Observations: (batch_size, stack, height, width, channels) -> (batch_size, stack*channels, height, width)
            self.observations = self.observations.permute(0, 1, 4, 2, 3).flatten(1, 2)
            self.next_observations = self.next_observations.permute(0, 1, 4, 2, 3).flatten(1, 2)
        else:
            assert self.observations.shape[1:] == (stack, image_size, image_size)
            assert self.next_observations.shape[1:] == (stack, image_size, image_size)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, index):
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "next_observations": self.next_observations[index],
            "dones": self.dones[index],
        }