#!/bin/bash
#SBATCH --job-name=mbtc3
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/mbart/cross/r3/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/mbart/cross/r3/err-%j
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
    --tgt_lang de \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy  epoch \
    --num_train_epochs  10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --train_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/train.csv \
    --validation_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/val.csv \
    --test_file /hits/basement/nlp/fatimamh/inputs/wiki_t5/cross/test.csv \
    --output_dir /hits/basement/nlp/fatimamh/outputs/mbart/cross/r3/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --overwrite_output_dir

    