import torch
from torch import nn, Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce



def discretize(
    t: Tensor,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)    # cube normalize
    t *= num_discrete
    t -= 0.5
    return t.round().long().clamp(min = 0, max = num_discrete - 1)



def undiscretize(
    t: Tensor,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = t.float()
    t += 0.5
    t /= num_discrete       # cube normalize
    return t * (hi - lo) + lo



class MeshTokenizer(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.pad_id = -1
        self.num_discrete_coors = args.n_discrete_size  # default: 800
        self.codebook_size = args.n_discrete_size       # default: 128
        self.coor_continuous_range = (-1., 1.)
    
    
    def tokenize(self, data_dict: dict) -> dict:
        '''
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>
        '''
        
        ### 3D mesh face parsing
        vertices = data_dict['vertices']    # batch x nv x 3
        faces = data_dict['faces']          # batch x nf x 3
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')   # batch x nf
        
        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        # fill padding tokens with 0, to prevent gather idx error
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        
        # collect vertice coordinates per-face: b x nf x nv x c
        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)
        face_coords = vertices.gather(-2, faces_vertices.long())
        
        # continuous to discrete face coords: b x nf x nv x c
        discrete_face_coords = discretize(
            face_coords,
            continuous_range=self.coor_continuous_range,
            num_discrete=self.num_discrete_coors
        )
        
        # pad invalid faces with <pad_id>: batch x nf x nv x c
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            self.pad_id
        )
        
        
        ### mesh to sequence convertion: batch x ntokens
        input_ids = discrete_padded_coords.reshape(batch, -1)
        attention_mask = (input_ids != self.pad_id).float()
        # reserve two spots:
        #     input_ids: <bos> ... <eos> <pad> ... => <pad> ... <pad> <pad> ...
        #     attn_mask:    1  ...    1     0  ... =>    1  ...    1     0  ...
        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        input_ids = torch.cat((place_holder * self.pad_id, input_ids, place_holder * self.pad_id), dim=1)
        attention_mask = torch.cat((place_holder, place_holder, attention_mask), dim=1)
        
        ### meshXL inputs
        data_dict['input_ids'] = input_ids.long()               # batch x (nf * 3 * 3 + 2)
        data_dict['attention_mask'] = attention_mask.float()    # batch x (nf * 3 * 3 + 2)
        
        # discard <bos> and <eos> tokens
        data_dict['codes'] = discrete_padded_coords.long()      # batch x (nf * 3 * 3)
        data_dict['discrete_face_coords'] = discrete_face_coords
        
        return data_dict
    
    
    def detokenize(self, input_ids: Tensor) -> dict:
        '''
        Turn sequential tokens: <bos> [<x>, <y>, <z>], ... <eos> into 3D meshes
        '''
        # input_ids: b (n q) or b n q, without <bos> or <eos>
        input_ids = input_ids.reshape(input_ids.shape[0], -1)
        # batch x nface
        face_mask = reduce(
            input_ids != self.pad_id, 'b (nf c) -> b nf', 'all', c = 9
        )
        
        # batch x (nface x 9) -> batch x nface x 3 x 3
        pred_face_coords = input_ids.reshape(input_ids.shape[0], -1, 9)
        pred_face_coords = rearrange(
            pred_face_coords, '... (v c) -> ... v c', v = 3
        )
        
        # back to continuous space
        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )
        # mask padding coordinates out with nan
        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            float('nan')
        )
        output_dict = {}
        output_dict['recon_faces'] = continuous_coors
        
        return output_dict
    
    
    def forward(self, data_dict: dict) -> dict:
        
        encoder_output = self.tokenize(data_dict)
        decoder_output = self.detokenize(
            input_ids = encoder_output['codes'], 
        )
        data_dict.update(encoder_output)
        data_dict.update(decoder_output)
        return data_dict
