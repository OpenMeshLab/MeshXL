accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision bf16 \
    sample_t2m.py \
    --test_ckpt mesh-xl/x-mesh-xl-350m/pytorch_model.bin \
    --text '3d model of a table' \
    --top_k 25 \
    --top_p 0.95 \
    --temperature 0.1