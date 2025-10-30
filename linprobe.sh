EXP_NAME="8gpu_u_mae_linear_eval"

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 /home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/main_linprobe.py \
    --accum_iter 4 \
    --batch_size 256 \
    --model vit_base_patch16 --cls_token\
    --finetune /home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/train_output/8gpu_u_mae_pretrain/checkpoint-199.pth \
    --epochs 90 \
    --blr 0.1  \
    --weight_decay 0.0 \
    --log_dir /home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/train_logs/$EXP_NAME \
    --dist_eval --data_path data \
    --use_hf_dataset \
    --nb_classes 100 \
    --output_dir "/home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/train_output/$EXP_NAME" \
    --eval_test
