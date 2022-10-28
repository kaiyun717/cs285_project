import os
from os import path
from PIL import Image
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import pandas as pd
import pickle
import torch

torch.set_default_dtype(torch.float64)

class PlanarDataset(Dataset):
    width = 40
    height = 40
    action_dim = 2

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

        self.obs_image = True

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((PlanarDataset.width,
                                   PlanarDataset.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control']),
                                  self._process_image(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

    def return_obs_image(self) -> bool:
        return self.obs_image


class GymPendulumDatasetV2(Dataset):
    width = 48 * 2
    height = 48
    action_dim = 1

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

        self.obs_image = True

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((GymPendulumDatasetV2.width,
                                   GymPendulumDatasetV2.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control']),
                                  self._process_image(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)
    
    def return_obs_image(self) -> bool:
        return self.obs_image


class MujocoDataset(Dataset):

    def __init__(self, dir, stack=1):
        """
        dir: 
        stack (int): For image obs, stack != 1. 
                    For serial obs, stack = 1, i.e., 'processed_1.pkl'
        """
        self.dir = dir
        self.stack = stack  # NOTE: yes we do
        self.dataframe: pd.DataFrame = pd.read_pickle(dir + '/dataframe.pkl')
        
        # Determine obs type : image or serial
        self.obs_image = False             # Serial
        for fname in os.listdir(dir):
            if (fname.endswith('.jpg') or fname.endswith('.png')):
                self.obs_image = True      # Image
                break
        
        if self.obs_image:
            assert stack != 1, "For image obs, you need to stack states."
        else:
            assert stack == 1, "For serial obs, you cannot stack states."

        self._processed = []
        self._process()

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        processed_img = ToTensor()((img.convert('L')))
        
        # print("Processed Image: ")
        # print(processed_img.shape)

        return processed_img   # No need for resizing!
                               # Downsampling happens when collecting.
    def _process(self):
        file_name = 'processed_{}.pkl'.format(str(self.stack))
        preprocessed_file = os.path.join(self.dir, file_name)
        if not os.path.exists(preprocessed_file):
            if self.obs_image:
                processed: list = self._process_obs_image()
            else:
                processed: list = self._process_obs_serial()
            
            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

    def _process_obs_image(self) -> list:
        processed = []
        print(self.stack)
        print(len(self))
        for i in trange(self.stack-1, len(self), 
                        desc='processing image data'):

            before = []
            after = []
            for t in reversed(range(self.stack)):
                # i = 3
                # t = 3,2,1,0
                # --> i - t = 0, 1, 2, 3

                # len = 100
                # i = 96, 97, 98, 99 
                b_idx = i - t       # before
                # a_idx = i - t + 1   # after
                temp_before = Image.open(os.path.join(self.dir, 
                                            f'before-{str(b_idx).zfill(5)}.jpg'))
                temp_after = Image.open(os.path.join(self.dir, 
                                            f'after-{str(b_idx).zfill(5)}.jpg'))
                before.append(self._process_image(temp_before))
                after.append(self._process_image(temp_after))

            # img_num = str(i).zfill(5)
            # before = Image.open(os.path.join(self.dir, f'before-{img_num}.jpg'))
            # after = Image.open(os.path.join(self.dir, f'after-{img_num}.jpg'))

            processed.append((torch.cat(tuple(before)),
                              np.array(self.dataframe.loc[i, 'action']),
                              torch.cat(tuple(after))))
        return processed

    def _process_obs_serial(self) -> list:
        processed = []
        for i in trange(len(self), desc='processing serial data'):
            before = np.array(self.dataframe.loc[i, 'before'])
            control = np.array(self.dataframe.loc[i, 'action'])
            after = np.array(self.dataframe.loc[i, 'after'])

            processed.append((before, control, after))

        return processed
    
    def return_obs_image(self) -> bool:
        return self.obs_image

