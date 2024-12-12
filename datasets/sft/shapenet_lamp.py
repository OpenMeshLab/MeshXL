import os
import numpy as np
from tqdm import tqdm
from eval_utils.perplexity import evaluate
from datasets.sft.base_dataset import BaseDataset, BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, 'MeshXL-shapenet-data')



class Dataset(BaseDataset):
    
    def __init__(self, args, split_set="train", augment=False): 
        super().__init__()
        
        # base dataset config
        self.dataset_name = 'shapenet_lamp'
        self.category_id = '03636649'
        self.eval_func = evaluate
        self.augment = augment and (split_set == 'train')
        self.num_repeat = 1
        self.pad_id = -1
        self.max_triangles = args.n_max_triangles
        self.max_vertices = self.max_triangles * 3
        
        # pre-load data into memory
        full_data = []
        for filename in tqdm(os.listdir(DATASET_DIR)):
            if self.category_id not in filename:
                continue
            if (split_set in filename) and filename.endswith('.npz'):
                loaded_data = np.load(
                    os.path.join(DATASET_DIR, filename),
                    allow_pickle=True
                )
                loaded_data = loaded_data["arr_0"].tolist()
                loaded_data = self._preprocess_data(loaded_data)
                full_data = full_data + loaded_data
        
        self.data = full_data
        
        print(f"[MeshDataset] Created from {len(self.data)} shapes for {self.dataset_name} {split_set}")
