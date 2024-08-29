import os
import torch
import argparse
import numpy as np
from torch import Tensor
from accelerate import Accelerator
from transformers import AutoTokenizer
from utils.ply_helper import write_ply
from models.x_mesh_xl.get_model import get_model


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


def make_args_parser():
    parser = argparse.ArgumentParser(
        "MeshXL: Neural Coordinate Field for Generative 3D Foundation Models", 
        add_help=False
    )
    ##### model config #####
    parser.add_argument("--llm", default='mesh-xl/mesh-xl-350m', type=str)
    parser.add_argument("--n_discrete_size", default=128, type=int)
    parser.add_argument("--text_condition", default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument("--test_ckpt", default='mesh-xl/x-mesh-xl-350m/pytorch_model.bin', type=str)
    parser.add_argument("--text", default='3d model of a chair', type=str)
    parser.add_argument("--output_dir", default='outputs', type=str)
    parser.add_argument("--num_samples", default=4, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--temperature", default=0.1, type=float)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = make_args_parser()
    accelerator = Accelerator()

    # prepare model
    tokenizer = AutoTokenizer.from_pretrained(args.text_condition)
    mesh_xl = get_model(args)
    mesh_xl.load_state_dict(torch.load(args.test_ckpt, map_location='cpu'))
    mesh_xl = accelerator.prepare(mesh_xl)

    net_device = next(mesh_xl.parameters()).device
    num_samples = args.num_samples

    text_inputs = tokenizer.batch_encode_plus(
        [args.text],
        max_length=64, 
        padding='max_length', 
        truncation='longest_first', 
        return_tensors='pt'
    )
    text_inputs = dict(
        text_input_ids=text_inputs['input_ids'].to(net_device),
        text_attention_mask=text_inputs['attention_mask'].to(net_device)
    )

    # model forward
    output_dict = mesh_xl(
        text_inputs,
        is_eval=True, 
        is_generate=True,
        num_return_sequences=args.num_samples,
        generation_config=dict(
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature
        )
    )

    # save samples
    os.makedirs(args.output_dir, exist_ok=True)
    for gen_id, sample in enumerate(output_dict['recon_faces']):
        post_process_mesh(
            sample, 
            os.path.join(
                args.output_dir,
                f'{accelerator.process_index:04d}_{gen_id}.ply'
            )
        )
