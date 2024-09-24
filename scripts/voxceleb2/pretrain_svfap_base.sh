# Set ouput directory
pretrain_dataset='voxceleb2'
model_dir="svfap_pretrain_base_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100" # directory of the pretrained model
OUTPUT_DIR="./saved/model/pretraining/${pretrain_dataset}/${model_dir}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
# Set the path to pre-training dataset.
DATA_PATH='./saved/data/voxceleb2/info_clean.csv'
# batch_size can be adjusted according to number of GPUs
# this script is for 4 GPUs (1 nodes x 4 GPUs)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --input_size 160 \
        --model pretrain_svfap_base_patch16_160 \
        --decoder_depth 4 \
        --batch_size 64 \
        --num_samples 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --stage_depths 12 6 3 \
        --temporal_kernel_size 2 \
        --num_bottleneck_tokens 8 \
        --num_workers 16 \
#        >${OUTPUT_DIR}/nohup.out 2>&1 &

