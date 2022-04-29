#!/bin/bash
#SBATCH --job-name=mt5c
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/mt5/cross/r2/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/mt5/cross/r2/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.0.130

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate py_lite

python run_mt5.py \
 	--resume_from_checkpoint  True \
    --model_name_or_path /hits/basement/nlp/fatimamh/outputs/mt5/cross/r2/checkpoint-102630 \
    --tokenizer_name /hits/basement/nlp/fatimamh/outputs/mt5/cross/r2/checkpoint-102630 \
    --do_predict \
    --test_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/test.csv \
    --output_dir /hits/basement/nlp/fatimamh/outputs/mt5/cross/r2/ \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --overwrite_output_dir \
