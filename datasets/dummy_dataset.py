import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from eval_utils.sample_generation import evaluate

class Dataset:
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.eval_func = evaluate
        
    def __len__(self):
        return 10
    
    def __getitem__(self, idx): 
        data_dict = {}
        data_dict['shape_idx'] = np.asarray(idx).astype(np.int64)
        return data_dict
