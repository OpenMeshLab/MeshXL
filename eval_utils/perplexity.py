import os
import time
import torch
import numpy as np
from torch import nn, Tensor
from collections import defaultdict, OrderedDict
from utils.ply_helper import write_ply
from utils.misc import SmoothedValue
from accelerate.utils import set_seed



def perplexity(neg_log_likelihood: list) -> Tensor:
    # gather per-sequence log likelihood for perplexity
    nll_chunk = torch.cat(neg_log_likelihood, dim=0)
    return torch.exp(nll_chunk.mean())



def post_process_mesh(mesh_coords: Tensor, filename: str):
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
    logger,
    curr_train_iter=-1,
):
    
    model.eval()
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)
    
    ### parse evaluation status
    if hasattr(dataset_loader.dataset, "dataset_name"):
        dataset_name = dataset_loader.dataset.dataset_name
    else:
        dataset_name = "default"
    task_name_prefix = dataset_name + '_'

    time_delta = SmoothedValue(window_size=10)
    
    accelerator.wait_for_everyone()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    if accelerator.is_main_process:
        logger.log_messages("==" * 10)
        logger.log_messages(f"Evaluate Epoch [{curr_epoch}/{args.max_epoch}]")
        logger.log_messages("==" * 10)
    
    ### calculate perplexity
    neg_log_likelihood = []
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        
        # forward pass to calculate per-sequence negative log likelihood
        with accelerator.autocast():    
            outputs = model(batch_data_label, is_eval=True)
        # [(batch,), (batch,), ...]
        neg_log_likelihood.append(outputs['neg_log_likelihood'])

        ### log status
        time_delta.update(time.time() - curr_time)
        
        if accelerator.is_main_process and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            moving_average_ppl = perplexity(neg_log_likelihood)
            logger.log_messages(
                '; '.join(
                    (
                        f"Evaluate {epoch_str}",
                        f"Batch [{curr_iter}/{num_batches}]",
                        f"perplexity: {moving_average_ppl:0.4f}",
                        f"Evaluating on iter: {curr_train_iter}",
                        f"Iter time {time_delta.avg:0.2f}",
                        f"Mem {mem_mb:0.2f}MB",
                    )
                )
            )
        
        ### end of an iteration
        
    ### end of a round
    
    quantitative = {
        task_name_prefix + 'ppl': perplexity(neg_log_likelihood).item()
    }
    
    ### do sampling every evaluation
    curr_time = time.time()
    
    set_seed(accelerator.process_index)
    
    # create storage directory
    storage_dir = os.path.join(args.checkpoint_dir, task_name_prefix + 'visualization')
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok = True)
    accelerator.wait_for_everyone()
    
    # just sample one round for checking training status
    for round_idx in range(1):
        outputs = model(
            data_dict=dict(),
            is_eval=True, 
            is_generate=True,
            num_return_sequences=args.batchsize_per_gpu,
        )
    
        generated_meshes = outputs["recon_faces"]   # nsample x nf x 3 x 3
    
        for sample_idx in range(args.batchsize_per_gpu):
            # store the generated meshes
            post_process_mesh(
                generated_meshes[sample_idx],
                os.path.join(
                    storage_dir, 
                    '_'.join(
                        (
                            f'{accelerator.process_index:04d}',
                            f'{round_idx:04d}',
                            f'{sample_idx:04d}.ply',
                        )
                    )
                )
            )
    
    accelerator.wait_for_everyone()
    
    return {}, quantitative