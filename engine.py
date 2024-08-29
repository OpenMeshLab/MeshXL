import time, math
import torch
import datetime

from utils.io import save_checkpoint
from utils.misc import SmoothedValue



def compute_learning_rate(args, curr_iter, max_iters):
    assert curr_iter <= max_iters and curr_iter >= 0
    if (curr_iter <= args.warm_lr_iters) and args.warm_lr_iters > 0:
        # Linear Warmup: warm_lr -> curr_lr -> base_lr
        curr_lr = args.warm_lr + curr_iter / args.warm_lr_iters * (args.base_lr - args.warm_lr)
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_iter / max_iters)
        )
    return curr_lr



def adjust_learning_rate(args, optimizer, curr_iter, max_iters):
    curr_lr = compute_learning_rate(args, curr_iter, max_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr



def do_train(
    args,
    model,
    accelerator,
    optimizer,
    dataloaders,
    best_val_metrics,
    logger
):
    
    if accelerator.is_main_process:
        logger.log_messages(f"call with args: {args}")
        logger.log_messages(f"{model}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    
    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    loss_break_down_avg = {}
    
    model.train()
    accelerator.wait_for_everyone()
    
    for curr_epoch in range(args.start_epoch, args.max_epoch):
        
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            
            curr_time = time.time()
            
            ### core for model training
            
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter, max_iters)
            
            with accelerator.accumulate(model):
                
                with accelerator.autocast():
                    outputs = model(batch_data_label)
                loss = outputs['loss']
                
                # sanity check, skip the infinite loss
                if not math.isfinite(loss.item()):
                    logger.log_messages("Loss in not finite. Skip this iteration.")
                    model.eval()
                    model.train()
                    torch.cuda.empty_cache()
                    continue
                
                accelerator.backward(loss)
                if args.clip_gradient > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_gradient)
                
                optimizer.step()
                optimizer.zero_grad()
            
            ### logging training loss status
            
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
            
            for key, value in outputs.items():
                if 'loss' in key.lower():
                    loss_break_down_avg[key] = loss_break_down_avg.get(key, SmoothedValue(window_size=10))
                    loss_break_down_avg[key].update(value.item())
    
            ### writing logs
            
            if accelerator.is_main_process and curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                logger.log_messages(
                    '; '.join(
                        [
                            f"Epoch [{curr_epoch}/{args.max_epoch}]",
                            f"Iter [{curr_iter}/{max_iters}]",
                            # loss string
                            *(
                                f'{key} {avg.avg:0.4f}' \
                                    for key, avg in loss_break_down_avg.items()
                            ),
                            # status string
                            f"LR {curr_lr:0.2e}",
                            f"Iter time {time_delta.avg:0.2f}s",
                            f"ETA {eta_str}",
                            f"Mem {mem_mb:0.2f}MB"
                        ]
                    )
                )
                train_loss_log = {k: v.avg for k, v in loss_break_down_avg.items()}
                train_loss_log["learning_rate"] = curr_lr
                logger.log_scalars(train_loss_log, prefix='train', step=curr_iter)
            
            ### saving checkpoints
            
            if accelerator.is_main_process and (curr_iter + 1) % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    accelerator.unwrap_model(model),
                    optimizer,
                    curr_epoch,
                    args,
                    best_val_metrics,
                    filename=f"checkpoint_{(curr_iter + 1) // 1000}k.pth",
                )
            
            ### pending and doing evaluations: every xxx after xxx iterations
            
            do_eval_flag = (curr_iter + 1) % args.eval_every_iteration == 0
            do_eval_flag &= (curr_iter + 1) > args.start_eval_after
            do_eval_flag |= (curr_iter + 1) == max_iters
            
            if do_eval_flag is True:
                eval_metrics = {}
                model.eval()
                with accelerator.autocast():
                    for test_loader in dataloaders['test']:
                        task_metrics, eval_loss_dict = test_loader.dataset.eval_func(
                            args,
                            curr_epoch,
                            accelerator.unwrap_model(model),
                            accelerator,
                            test_loader,
                            logger,
                            curr_train_iter=curr_iter
                        )
                        eval_metrics.update(task_metrics)
                        logger.log_scalars(eval_loss_dict, prefix='val', step=curr_iter)
                model.train()
                
                ### saving `checkpoint_best.pth` do nothing for unknown criterion
                
                if args.criterion is None:
                    continue
                
                if not best_val_metrics or (
                    best_val_metrics[args.criterion] < eval_metrics[args.criterion]
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        accelerator.unwrap_model(model),
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    if accelerator.is_main_process:
                        logger.log_messages(
                            f"Epoch [{curr_epoch}/{args.max_epoch}] "
                            f"saved current best val checkpoint at {filename}; "
                            f"{args.criterion} {eval_metrics[args.criterion]}"
                        )
            
            ### end of an iteration
            
        ### end of an epoch
        
        save_checkpoint(
            args.checkpoint_dir,
            accelerator.unwrap_model(model),
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
    
    # end of training
    
    return 
