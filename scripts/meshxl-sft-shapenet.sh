export BASE_MESHXL=meshxl/mesh-xl-1.3b
export BATCHSIZE_PER_GPU=2

accelerate launch \
    --config_file ./config/deepspeed_stage2.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    main.py \
    --dataset sft.shapenet_table \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --warm_lr_iters -1 \
    --base_lr 1e-6 \
    --llm $BASE_MESHXL \
    --model mesh_xl \
    --checkpoint_dir ./ckpts/mesh_xl_1.3b_base_pretrain_bs2_8a100 \
    --batchsize_per_gpu $BATCHSIZE_PER_GPU \
    --dataset_num_workers 0 \
    --augment \
    --eval_every_iteration 10000 \
    --save_every 20000 \
    --max_epoch 1024