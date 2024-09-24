# pre-training parameters
pretrain_dataset='voxceleb2'
model_dir="svfap_pretrain_base_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100" # directory of the pretrained model
ckpt=99 # which pretrained ckpt
# fine-tuning parameters
finetune_dataset='mafw'
num_labels=11
input_size=160
sr=4
lr=1e-3
epochs=100
batch_size=16

splits=(1 2 3 4 5)
for split in "${splits[@]}";
do
  # output directory: save ckpt and log
  OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_split0${split}_lr${lr}_e${epochs}_size${input_size}_sr${sr}"
  if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
  fi
  # path to split files (train.csv/val.csv/test.csv)
  DATA_PATH="./saved/data/${finetune_dataset}/single/split0${split}"
  # path to pre-trained model
  MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/${model_dir}/checkpoint-${ckpt}.pth"

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 \
      --master_port 13590 \
      run_class_finetuning.py \
      --model svfap_base_patch16_${input_size} \
      --data_set ${finetune_dataset^^} \
      --nb_classes ${num_labels} \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size ${batch_size} \
      --num_sample 1 \
      --input_size ${input_size} \
      --short_side_size ${input_size} \
      --save_ckpt_freq 1000 \
      --num_frames 16 \
      --sampling_rate ${sr} \
      --opt adamw \
      --lr ${lr} \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs ${epochs} \
      --dist_eval \
      --test_num_segment 2 \
      --test_num_crop 2 \
      --stage_depths 12 6 3 \
      --temporal_kernel_size 2 \
      --num_bottleneck_tokens 8 \
      --num_workers 16 \
      >${OUTPUT_DIR}/nohup_rerun.out 2>&1
done
echo "Done!"



