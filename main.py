import os, argparse, importlib
import numpy as np
import torch

from engine import do_train
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn
from utils.logger import Logger


def make_args_parser():
    
    parser = argparse.ArgumentParser(
        "MeshXL: Neural Coordinate Field for Generative 3D Foundation Models", 
        add_help=False
    )

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_iters", default=1000, type=int)
    
    ##### Dataset Setups #####
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--dataset", default='shapenet_chair', help="dataset list split by ','")
    parser.add_argument("--augment", default=False, action='store_true', help="whether use data augmentation")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="discretized 3D space")
    parser.add_argument("--n_max_triangles", default=800, type=int, help="max number of triangles")
    
    ##### Model Setups #####
    parser.add_argument(
        '--model', 
        default=None, 
        type=str, 
        help="The model folder: unconditional / conditional mesh generation"
    )
    parser.add_argument(
        '--llm', 
        default=None, 
        type=str, 
        help="The LLM super config and pre-trained weights"
    )
    # conditonal mesh generation, set to None for unconditional generation
    parser.add_argument('--text_condition', default=None, type=str, help="the conditional language model")
    parser.add_argument('--image_condition', default=None, type=str, help="the conditional vision model")
    parser.add_argument('--pretrained_weights', default=None, type=str, help='checkpoint to pre-trained weights')
    parser.add_argument("--dataset_num_workers", default=4, type=int, help='number of workers for dataloader')
    parser.add_argument("--batchsize_per_gpu", default=8, type=int, help='batch size for each GPU')
    
    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int, help='overwrite by pre-trained weights')
    parser.add_argument("--max_epoch", default=16, type=int, help='number of traversals for the dataset')
    parser.add_argument("--start_eval_after", default=-1, type=int, help='do not evaluate the model before xxx iterations')
    parser.add_argument("--eval_every_iteration", default=4000, type=int, help='do evaluate the model every xxx iterations')
    parser.add_argument("--seed", default=0, type=int, help='random seed')

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--sample_rounds", default=100, type=int, help='do sample for xxx rounds to produce 3D meshes')

    parser.add_argument(
        "--criterion", default=None, type=str,
        help='metrics for saving the best model, set to None for not saving any'
    )
    parser.add_argument("--test_ckpt", default="", type=str, help='test checkpoint directory')

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str, help='path to save the checkpoints and visualization samples')
    parser.add_argument("--save_every", default=20000, type=int, help='save checkpoints every xxx iterations')
    parser.add_argument("--log_every", default=10, type=int, help='write training logs every xxx iterations')
    
    args = parser.parse_args()
    
    return args


def build_dataloader_func(args, dataset, split):
    
    if split == "train":
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
        # add for meshgpt
        drop_last = True,
    )
    return sampler, dataloader


def build_dataset_func(args):
    
    datasets = {
        'train': [], 
        'test': []
    }
    
    for dataset in args.dataset.split(','):
        dataset_module = importlib.import_module(f'datasets.{dataset}')
        datasets['train'].append(
            dataset_module.Dataset(args, split_set="train", augment=args.augment)
        )
        datasets['test'].append(
            dataset_module.Dataset(args, split_set="val", augment=False)
        )
    datasets['train'] = torch.utils.data.ConcatDataset(datasets['train'])
    
    train_sampler, train_loader = build_dataloader_func(args, datasets['train'], split='train')
    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }
    
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)
    
    return datasets, dataloaders    


def build_model_func(args):
    model_module = importlib.import_module(f'models.{args.model}.get_model')
    model = model_module.get_model(args)
    return model


def main(args):
    
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        # if not define the checkpoint-dir, set to the test checkpoint folder as default
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!'
        )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    accelerator = Accelerator()
    set_seed(args.seed)
    
    ### build datasets and dataloaders
    datasets, dataloaders = build_dataset_func(args)
    
    ### build models
    model = build_model_func(args)
    ### set default checkpoint
    checkpoint = None
    
    # testing phase
    if args.test_only:
        
        try:
            checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model"], strict=False)
        except:
            print('test the model from scratch...')
        
        model, dataloaders['train'], *dataloaders['test'] = accelerator.prepare(
            model, dataloaders['train'], *dataloaders['test']
        )
        
        for test_loader in dataloaders['test']:
            test_loader.dataset.eval_func(
                args,
                -1,
                model,
                accelerator,
                test_loader
            )
        
    # training phase
    else:
        
        assert (
            args.checkpoint_dir is not None
        ), "`--checkpoint_dir` is required to identify the directory to store the checkpoint"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        logger = Logger(args.checkpoint_dir, accelerator)
        
        ### whether or not use pretrained weights
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            model.load_state_dict(checkpoint['model'], strict=False)
            
        if accelerator.is_main_process:
            if checkpoint is not None:
                logger.log_messages('Loaded the parameters for weight initialization:')
                for name, param in checkpoint['model'].items():
                    logger.log_messages('\t'.join(['', name + ':', f'{param.shape}']))
                logger.log_messages('\n' * 10)
                logger.log_messages('====\n')
                logger.log_messages('The trainable parameters are:')
            
            for name, param in model.named_parameters():
                status = '[train]' if param.requires_grad else '[eval]'
                logger.log_messages('\t'.join(['', status, name + ':', f'{param.shape}']))
                        
        optimizer = torch.optim.AdamW(
            filter(lambda params: params.requires_grad, model.parameters()), 
            lr=args.base_lr, 
            weight_decay=args.weight_decay
        )
        
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        
        model, optimizer, dataloaders['train'], *dataloaders['test'] = accelerator.prepare(
            model, optimizer, dataloaders['train'], *dataloaders['test']
        )
        
        do_train(
            args,
            model,
            accelerator,
            optimizer,
            dataloaders,
            best_val_metrics,
            logger
        )


if __name__ == "__main__":
    args = make_args_parser()
    
    os.environ['PYTHONWARNINGS']='ignore:semaphore_tracker:UserWarning'

    main(args)