import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from einops import repeat, rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, Blip2QFormerModel, Blip2QFormerConfig
from models.x_mesh_xl.tokenizer import MeshTokenizer
from typing import Dict


class ConditionEncoder(nn.Module):

    def train(self, mode: bool = True):
        super().train(mode)
        self.multi_encoder.eval()
        for param in self.multi_encoder.parameters():
            param.requires_grad = False
        return self

    def __init__(self, args, hidden_size):
        super().__init__()
        self.n_learnable_queries = 32
        config = AutoConfig.from_pretrained(args.text_condition)
        self.multi_encoder = AutoModel.from_config(config)
        qformer_config = Blip2QFormerConfig(
            num_hidden_layers=12,
            encoder_hidden_size=self.multi_encoder.config.text_config.hidden_size
        )
        self.qformer = Blip2QFormerModel(qformer_config)
        self.query_embeds = nn.Embedding(self.n_learnable_queries, qformer_config.hidden_size)
        self.out_project = nn.Linear(qformer_config.hidden_size, hidden_size)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        text_encoder_output = self.multi_encoder.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_encoder_output.last_hidden_state
        return text_embeds      # bs x ntoken x ch

    def forward(self, input_ids, attention_mask):
        net_device = next(self.parameters()).device
        text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        query_embeds = self.query_embeds(
            repeat(
                torch.arange(0, self.n_learnable_queries, dtype=torch.int64).to(net_device),
                'src -> bs src',
                bs = text_embeds.shape[0]
            )
        )
        query_outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=attention_mask
        )
        query_outputs = query_outputs[0][:, : self.n_learnable_queries, :]
        return self.out_project(query_outputs)



class MeshXL(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        # self.transformer.eval()
        # for param in self.transformer.parameters():
        #     param.requires_grad = False
        return self
    
    def __init__(self, args):
        super().__init__()
        
        self.tokenizer = MeshTokenizer(args)
        
        # causal LM model initialization
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2
        
        config = AutoConfig.from_pretrained(
            args.llm, 
            n_positions=8192,
            max_position_embeddings=8192,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )

        config.word_embed_proj_dim = config.hidden_size
        self.transformer = AutoModelForCausalLM.from_config(config=config)
        
        try:
            self.transformer.to_bettertransformer()
        except:
            pass
        
        self.condition_encoder = ConditionEncoder(args, config.hidden_size)

        # setting status for all parameters
        self.train()
    
    
    def forward(
            self, 
            data_dict: dict=None, 
            is_eval: bool=False, 
            is_generate: bool=False,
            num_return_sequences: int=8, 
            generation_config: Dict=dict(
                do_sample=True,
                top_k=50,
                top_p=0.95,
                # no_repeat_ngram_size=9,
            )
        ) -> dict:
        
        data_dict['prefix_embeds'] = self.condition_encoder(
            input_ids = data_dict['text_input_ids'],
            attention_mask = data_dict['text_attention_mask']
        )

        if not is_eval:
            return NotImplementedError
        
        if is_eval and not is_generate:
            return NotImplementedError
        
        if is_eval and is_generate:
            return self.generate(
                data_dict=data_dict, 
                num_return_sequences=num_return_sequences, 
                generation_config=generation_config
            )
        
        raise NotImplementedError('training status undefined!')
        return 
    
    @torch.no_grad()
    def generate(self, data_dict: dict=None, num_return_sequences: int=8, generation_config: dict=dict()) -> dict:

        net_device = next(self.parameters()).device
        max_length = 8191
        output_ids = torch.ones(num_return_sequences, max_length).long().to(net_device) * self.eos_token_id
        
        # batch x ntokens
        results = self.transformer.generate(
            inputs_embeds=data_dict['prefix_embeds'],
            max_length=max_length-1,
            num_return_sequences=num_return_sequences,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
            **generation_config
        )
        output_ids[:, :results.shape[1]] = results
        
        # discard <bos> and <eos> tokens to pad tokens
        output_ids = output_ids[:, :-1]
        output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id
        
        decoder_output = self.tokenizer.detokenize(input_ids=output_ids)
        
        return decoder_output
        


def get_model(args):
    model = MeshXL(args)
    return model