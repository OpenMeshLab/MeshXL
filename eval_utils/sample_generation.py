import os
import time
import tqdm
import torch
import numpy as np
from torch import Tensor
from collections import defaultdict, OrderedDict
from utils.ply_helper import write_ply
from utils.misc import SmoothedValue
from accelerate.utils import set_seed


def process_mesh(mesh_coords: Tensor, filename: str):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)
    write_ply(
        np.asarray(vertices.cpu()),
        None,
        np.asarray(triangles),
        filename
    )
    return vertices


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    accelerator,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):

    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    storage_dir = os.path.join(args.checkpoint_dir, 'sampled')
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok = True)
    
    model.eval()
    accelerator.wait_for_everyone()
    
    # do sampling
    curr_time = time.time()
    
    set_seed(accelerator.process_index)

    for sample_round in tqdm.tqdm(range(args.sample_rounds)):

        outputs = model(None, n_samples=args.batchsize_per_gpu, is_eval=True, is_generate=True)

        batch_size = outputs['recon_faces'].shape[0]
        generated_faces = outputs["recon_faces"]
    
        for batch_id in range(batch_size):
            process_info = f'{accelerator.process_index:04d}'
            sample_info = f'{sample_round:04d}'
            batch_sample_info = f'{batch_id:04d}'
            sample_id = '_'.join(
                [
                    process_info,
                    sample_info,
                    batch_sample_info
                ]
            )
            process_mesh(
                generated_faces[batch_id],
                os.path.join(storage_dir, f'{sample_id}_generated.ply')
            )
    
    # Memory intensive as it gathers point cloud GT tensor across all ranks
    time_delta.update(time.time() - curr_time)
    accelerator.wait_for_everyone()
    
    return {}, {}