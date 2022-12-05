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

class OfflineDataset(Dataset):

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
        self.obs_image: bool = False             # Serial
        for fname in os.listdir(dir):
            if (fname.endswith('.jpg') or fname.endswith('.png')):
                self.obs_image = True      # Image
                break
        
        self._processed = []
        self._process()

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        processed_img = ToTensor()((img.convert('L')))
        return processed_img   # No need for resizing!
                               # Downsampling happens when collecting.
    def _process(self):
        file_name = 'processed_{}.pkl'.format(str(self.stack))
        preprocessed_file = os.path.join(self.dir, file_name)
        if not os.path.exists(preprocessed_file):
            processed: list = self._process_samples()
            
            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
                f.close()
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

    def _process_samples(self) -> list:
        if self.obs_image:
            desc = 'processing image data'
        else:
            desc = 'processing serial data'
        processed = []
        for i in trange(self.stack-1, len(self), 
                        desc=desc):

            before = []
            after = []
            for t in reversed(range(self.stack)):
                ##################################
                ##################################
                ##### i = 3                  #####
                ##### t = 3,2,1,0            #####
                ##### --> i - t = 0, 1, 2, 3 #####
                ##### len = 100              #####
                ##### i = 96, 97, 98, 99     #####
                ##################################
                ##################################
                idx = i - t       # idx

                # If `done=True` in midst of stacking, then skip to next.
                done = bool(self.dataframe.loc[idx, 'done'])
                if done and t != 0:
                    break
                
                if self.obs_image:  # saved as images
                    temp_before = Image.open(os.path.join(self.dir,
                                                f'before-{str(idx).zfill(6)}.jpg'))
                    temp_after = Image.open(os.path.join(self.dir,
                                                f'after-{str(idx).zfill(6)}.jpg'))
                    before.append(self._process_image(temp_before))
                    after.append(self._process_image(temp_after))
                else:               # saved as serial
                    before.append(torch.Tensor(self.dataframe.loc[idx, 'before']))
                    after.append(torch.Tensor(self.dataframe.loc[idx, 'after']))

            # If `done=True` in midst of stacking, then skip to next.
            if len(before) != self.stack:
                continue

            processed.append((torch.cat(tuple(before)),
                              np.array(self.dataframe.loc[i, 'action']),
                              torch.cat(tuple(after)),
                              np.array(self.dataframe.loc[i, 'reward']),
                              np.array(self.dataframe.loc[i, 'done'])))
        print('OFFLINE DATASET SIZE: ', len(processed))
        return processed

    def return_obs_image(self) -> bool:
        return self.obs_image
