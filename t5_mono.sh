#!/bin/bash
#SBATCH --job-name=t5-m
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/t5/mono/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/t5/mono/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.0.130

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate py_lite

python run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --train_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/mono/train.csv \
    --validation_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/mono/val.csv \
    --source_prefix "summarize: " \
    --output_dir /hits/basement/nlp/fatimamh/outputs/t5/mono/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
