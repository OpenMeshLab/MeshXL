# Copyright (c) Facebook, Inc. and its affiliates.
import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    
    def __init__(self, log_dir=None, accelerator=None) -> None:
        self.log_dir = log_dir
        self.accelerator = accelerator
        
        if self.log_dir is not None:
            self.txt_writer = open(os.path.join(self.log_dir, 'logger.log'), 'a')
        else:
            self.txt_writer = None
        
        if SummaryWriter is not None and self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def log_scalars(self, scalar_dict, step, prefix=None):
        if self.writer is None:
            return
        for k in scalar_dict:
            v = scalar_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if prefix is not None:
                k = prefix + '_' + k
            self.writer.add_scalar(k, v, step)
        return

    def log_messages(self, message: str):
        if self.txt_writer is not None:
            self.txt_writer.write(message + "\n")
            self.txt_writer.flush()
        print(message, flush=True)
        return
    