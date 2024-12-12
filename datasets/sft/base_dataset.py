import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from eval_utils.perplexity import evaluate

BASE_DIR = os.path.join('.', 'data')


def scale_mesh(vertices: np.ndarray, scale_factor: tuple=(0.75, 1.25)) -> np.ndarray:
    lower, upper = scale_factor
    scale_axis = lower + np.random.rand(vertices.shape[-1]) * (upper - lower)
    vertices = vertices * scale_axis
    return vertices


def normalize_mesh(vertices: np.ndarray, scale_range: tuple=(-1.0, 1.0)) -> np.ndarray:
    lower, upper = scale_range
    scale_per_axis = (vertices.max(0) - vertices.min(0)).max()
    center_xyz = 0.5 * (vertices.max(0) + vertices.min(0))
    normalized_xyz = (vertices - center_xyz) / scale_per_axis   # scaled into range (0, 1)
    vertices = normalized_xyz * (upper - lower)
    return vertices
    

class BaseDataset:
    
    def __init__(self, *args, **kwargs): 
        super().__init__()
        self.data =[]
        self.num_repeat = 1
        
    def _preprocess_data(self, data_chunk):
        processed = []
        for data in data_chunk:
            processed.append(
                dict(
                    vertices = np.asarray(data['vertices']),
                    faces = np.asarray(data['faces']),
                )
            )
        return processed

    def _fetch_data(self, idx):
        idx = idx % len(self.data)
        return deepcopy(self.data[idx])
        
    def __len__(self):
        return len(self.data) * self.num_repeat
    
    def __getitem__(self, idx):
        data = self._fetch_data(idx)
        data['vertices'] = np.asarray(data['vertices'])
        data['faces'] = np.asarray(data['faces'])
        
        num_vertices = len(data['vertices'])
        num_faces = len(data['faces'])
        
        vertices = np.ones((self.max_vertices, 3)) * self.pad_id
        faces = np.ones((self.max_triangles, 3)) * self.pad_id
        
        if self.augment is True:
            data['vertices'] = scale_mesh(data['vertices'])
        data['vertices'] = normalize_mesh(data['vertices'])
        
        vertices[:num_vertices] = data['vertices']
        faces[:num_faces] = data['faces']
        
        gt_vertices = vertices[faces.clip(0).astype(np.int64)]       # nface x 3 x 3
        gt_vertices[faces[:, 0] == self.pad_id] = float('nan')
        
        data_dict = {}
        data_dict['shape_idx'] = np.asarray(idx).astype(np.int64)
        data_dict['vertices'] = np.asarray(vertices).astype(np.float32)
        data_dict['faces'] = np.asarray(faces).astype(np.int64)
        data_dict['gt_vertices'] = np.asarray(gt_vertices).astype(np.float32)
        
        return data_dict  
