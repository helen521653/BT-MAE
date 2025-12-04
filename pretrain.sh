EXP_NAME="8gpu_fixed_data_and_masking_SMI_orig_mae_pretrain"

torchrun --nproc_per_node=8 /home/jovyan/lena/mi_mae_like/BT-MAE/mi_like_pretrain.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /tmp/datasets/imagenet100 \
    --lamb 0.00 \
    --reg none \
    --output_dir "/home/jovyan/lena/mi_mae_like/BT-MAE/train_output/$EXP_NAME" \
    --log_dir /home/jovyan/lena/mi_mae_like/BT-MAE/train_logs/${EXP_NAME} \
    --bt_variant per_image_cross \
    --bt_weight 0.005 \
    --bt_lambda 0.005
    # --distributed
