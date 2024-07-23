accelerate launch \
    --config_file ./config/deepspeed_stage2.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    main.py \
    --dataset pre_train.shapenet,pre_train.sketchfab \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --warm_lr_iters 1000 \
    --llm facebook/opt-1.3b \
    --model mesh_xl \
    --checkpoint_dir ./ckpts-base-models/mesh_xl_1.3b_base_pretrain_bs2_8a100 \
    --batchsize_per_gpu 2 \
    --dataset_num_workers 0 \
    --augment \
    --eval_every_iteration 10000 \
    --save_every 20000 \
    --max_epoch 8
    

sleep 1m

cd /mnt/share/cq8/sijinchen/get_gpu
python ssc.py
