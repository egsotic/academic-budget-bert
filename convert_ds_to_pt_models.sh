#! /bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export CUDA_VISIBLE_DEVICES=0

while getopts "d:t:V:v:r:" option; do
  case $option in
    d)
      dataset_name="$OPTARG"
      ;;
    t)
      tokenizer_type="$OPTARG"
      ;;
    V)
      vocab_size="$OPTARG"
      ;;
    v)
      version_name="$OPTARG"
      ;;
    r)
      current_run_id="$OPTARG"
      ;;

    *)
      echo "Usage: $0 [-d dataset_name] [-t tokenizer_type] [-V vocab_size] [-v version_name] [-r run_id]"
      exit 1
      ;;
  esac
done

tokenizer_name="${tokenizer_type}_${vocab_size}"

deepspeed convert_ds_to_pt_models.py \
          --model_type bert-mlm \
          --tokenizer_name "onlplab/alephbert-base" \
          --hidden_act gelu \
          --vocab_size $vocab_size \
          --hidden_size 1024 \
          --num_hidden_layers 24 \
          --num_attention_heads 16 \
          --intermediate_size 4096 \
          --hidden_dropout_prob 0.1 \
          --attention_probs_dropout_prob 0.1 \
          --encoder_ln_mode pre-ln \
          --lr 1e-3 \
          --train_batch_size 4096 \
          --train_micro_batch_size_per_gpu 16 \
          --lr_schedule step \
          --curve linear \
          --warmup_proportion 0.06 \
          --gradient_clipping 0.0 \
          --optimizer_type adamw \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.98 \
          --adam_eps 1e-6 \
          --max_steps 23000 \
          --total_training_time 9223372036854775807 \
          --early_exit_time_marker 9223372036854775807 \
          --dataset_path "/home/nlp/egsotic/data/heb_datasets/${dataset_name}/training_data_balanced/${tokenizer_name}/${version_name}" \
          --output_dir /home/nlp/egsotic/repo/academic-budget-bert/outputs/${dataset_name}_${tokenizer_name}_${version_name}-${current_run_id}/${current_run_id}/pt/ \
          --print_steps 100 \
          --num_epochs_between_checkpoints 100 \
          --job_name "convert_${dataset_name}_${tokenizer_name}_${version_name}" \
          --current_run_id "convert_${current_run_id}" \
          --project_name "budget-bert-pretraining" \
          --validation_epochs 50 \
          --validation_epochs_begin 1 \
          --validation_epochs_end 1 \
          --validation_begin_proportion 0.05 \
          --validation_end_proportion 0.01 \
          --validation_micro_batch 16 \
          --deepspeed \
          --data_loader_type dist \
          --do_validation \
          --seed 42 \
          --layer_norm_type pytorch \
          --load_training_checkpoint /home/nlp/egsotic/repo/academic-budget-bert/outputs/${dataset_name}_${tokenizer_name}_${version_name}-${current_run_id}/${current_run_id}/latest_checkpoint \
