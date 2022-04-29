#!/bin/bash
#SBATCH --job-name=mt5c
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/mt5/cross/r1/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/mt5/cross/r1/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.0.130

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate py_lite

python run_mt5.py \
    --model_name_or_path google/mt5-small \
    --tokenizer_name google/mt5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs  35 \
    --warmup_steps 100 \
    --train_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/train.csv \
    --validation_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/val.csv \
    --test_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/test.csv \
    --output_dir /hits/basement/nlp/fatimamh/outputs/mt5/cross/r1/ \
    --load_best_model_at_end True \
    --greater_is_better True \
    --metric_for_best_model "eval_bertscore_f" \
    --sortish_sampler True \
    --group_by_length True \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --logging_dir /hits/basement/nlp/fatimamh/outputs/mt5/cross/r1/logs/ \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --logging_steps 1 \
    --eval_steps 1 \
    --logging_first_step True \
    --save_strategy epoch \
    --save_total_limit 5 
    