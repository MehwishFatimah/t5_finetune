#!/bin/bash
#SBATCH --job-name=mbt-m
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/mbart/mono/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/mbart/mono/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.0.130

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate py_lite

python run_summarization.py \
    --model_name_or_path facebook/mbart-large-50 \
    --tokenizer_name facebook/mbart-large-50 \
    --src_lang en \
    --tgt_lang en \
    --do_train \
    --do_eval \
    --train_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/mono/train.csv \
    --validation_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/mono/val.csv \
    --output_dir /hits/basement/nlp/fatimamh/outputs/mbart/mono/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --overwrite_output_dir