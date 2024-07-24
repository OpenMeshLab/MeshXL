export LLM_CONFIG='mesh-xl/mesh-xl-1.3b'
export NSAMPLE_PER_GPU=2
export SAMPLE_ROUNDS=100
export OUTPUT_DIR='./output-samples-1.3b'

accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    main.py \
    --dataset dummy_dataset \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --sample_rounds $SAMPLE_ROUNDS \
    --dataset_num_workers 0 \
    --test_only
    