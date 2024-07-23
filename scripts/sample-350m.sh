export LLM_CONFIG='mesh-xl/mesh-xl-350m'
export NSAMPLE_PER_GPU=2
export SAMPLE_ROUNDS=100
export OUTPUT_DIR='./output-samples-350m'
export TEST_CKPT=''

accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    train_gpt.py \
    --dataset dummy_dataset \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --gpt_module mesh_xl \
    --test_ckpt $TEST_CKPT \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --sample_rounds $SAMPLE_ROUNDS \
    --dataset_num_workers 0 \
    --test_only
    
