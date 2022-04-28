#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

"""-------------------------------------------------------------------------------------------------------------
    
-------------------------------------------------------------------------------------------------------------"""

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

"""-------------------------------------------------------------------------------------------------------------
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
-------------------------------------------------------------------------------------------------------------"""
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},)
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},)
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},)
    use_auth_token: bool = field(default=False, metadata={"help": "Will use the token generated when running `login` (necessary to use this script with private models)."},)
    resize_position_embeddings: Optional[bool] = field(default=None, metadata={"help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                                                                                "the model's position embeddings."},)

"""-------------------------------------------------------------------------------------------------------------
    Arguments pertaining to what data we are going to input our model for training and eval.
-------------------------------------------------------------------------------------------------------------"""
@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json or csv file)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "The input evaluation data file to evaluate the metrics (rouge) on (a json or csv file)."},)
    test_file: Optional[str] = field(default=None, metadata={"help": "The input test data file to evaluate the metrics (rouge) on (a json or csv file)."},)
    
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."},)
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                                                                    "than this will be truncated, sequences shorter will be padded."},)
    max_target_length: Optional[int] = field(default=128, metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                                                                    "than this will be truncated, sequences shorter will be padded."},)
    val_max_target_length: Optional[int] = field(default=None, metadata={"help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                                                                        "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                                                                        "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                                                                        "during ``evaluate`` and ``predict``."},)

    pad_to_max_length: bool = field(default=True, metadata={"help": "Whether to pad all samples to model maximum sentence length."},)
    max_train_samples: Optional[int] = field(default=10, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this."},)
    max_eval_samples: Optional[int] = field(default=10, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this."},)
    max_predict_samples: Optional[int] = field(default=10, metadata={"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this."},)
    
    num_beams: Optional[int] = field(default=4, metadata={"help": "Number of beams to use for evaluation---used during ``evaluate`` and ``predict``."},)
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},)
    source_prefix: Optional[str] = field(default="summarize: ", metadata={"help": "A prefix to add before every source text."})

    forced_bos_token: Optional[str] = field(default=None, metadata={"help": "The token to force as the first generated token after the decoder_start_token_id."
                                                                    "Useful for multilingual models like mBART where the first generated token"
                                                                    "needs to be the target language token (Usually it is the target language token)"},)

    """-------------------------------------------------------------------------"""
    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


"""-------------------------------------------------------------------------------------------------------------
    Get predictions and labels
-------------------------------------------------------------------------------------------------------------"""
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

"""-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------"""
def main():

    """-------------------------------------------------------------------------"""
    #1. Setting data, model and training parameters.
    """-------------------------------------------------------------------------"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    """-------------------------------------------------------------------------"""
    #2. Setup logging
    """-------------------------------------------------------------------------"""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)],)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    """-------------------------------------------------------------------------"""
    #3. Log on each process the small summary:
    """-------------------------------------------------------------------------"""
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
                    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",]:
        logger.warning("You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with `--source_prefix 'summarize: ' `")

    """-------------------------------------------------------------------------"""
    #4. Detecting last checkpoint.
    """-------------------------------------------------------------------------"""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    """-------------------------------------------------------------------------"""
    #5. Set seed before initializing model for reproducibility
    """-------------------------------------------------------------------------"""
    set_seed(training_args.seed)

    """-------------------------------------------------------------------------"""
    #6. Load dataset 
    """-------------------------------------------------------------------------"""
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]

    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]

    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir,
                                use_auth_token=True if model_args.use_auth_token else None,)

    """-------------------------------------------------------------------------"""
    #7. Load the model config
    """-------------------------------------------------------------------------"""
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        cache_dir=model_args.cache_dir, revision=model_args.model_revision,
                                        use_auth_token=True if model_args.use_auth_token else None,)
    
    """-------------------------------------------------------------------------"""
    #8. Load the tokenizer
    """-------------------------------------------------------------------------"""
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                            cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
                                            revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,)
    
    """-------------------------------------------------------------------------"""
    #9. Load a Seq2Seq Model
    """-------------------------------------------------------------------------"""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                                config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, 
                                                use_auth_token=True if model_args.use_auth_token else None,)

    """-------------------------------------------------------------------------"""
    #10. Resize the model embeddings
    """-------------------------------------------------------------------------"""
    model.resize_token_embeddings(len(tokenizer))

    """-------------------------------------------------------------------------"""
    #11. Language and related settings for tokenizer
    """-------------------------------------------------------------------------"""
    src_lang = "en_XX"
    tgt_lang = "de_DE"

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    if (hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length):
        
        if model_args.resize_position_embeddings is None:
            logger.warning(f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                            f"to {data_args.max_source_length}.")
            model.resize_position_embeddings(data_args.max_source_length)
        
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        
        else:
            raise ValueError(f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`.")

    """-------------------------------------------------------------------------"""
    #12. Setting prefix
    """-------------------------------------------------------------------------"""
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    """-------------------------------------------------------------------------"""
    #13. Preprocessing the datasets. We need to tokenize inputs and targets.
    """-------------------------------------------------------------------------"""
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    """-------------------------------------------------------------------------"""
    #14. Language and related settings for tokenizer
    """-------------------------------------------------------------------------"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    forced_bos_token_id = (tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None)
    #forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    print("forced_bos_token_id: {}".format(forced_bos_token_id))
    model.config.forced_bos_token_id = forced_bos_token_id

    """-------------------------------------------------------------------------"""
    #15. Get the column names for input/target.
    """-------------------------------------------------------------------------"""
    dataset_columns = None
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    """-------------------------------------------------------------------------"""
    #16. Temporarily set max_target_length for training.
    """-------------------------------------------------------------------------"""
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    """-------------------------------------------------------------------------"""
    #17. Label smoothing setting
    """-------------------------------------------------------------------------"""
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning("label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

    """-------------------------------------------------------------------------------------------------------------
        Format model inputs
    -------------------------------------------------------------------------------------------------------------"""
    def preprocess_function(examples):

        #1. remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        #2. Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        #3. If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    
    """-------------------------------------------------------------------------"""
    #18. Training setting
    """-------------------------------------------------------------------------"""
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers,
                                            remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache,
                                            desc="Running tokenizer on train dataset",)

    """-------------------------------------------------------------------------"""
    #19. Evaluation setting
    """-------------------------------------------------------------------------"""
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers,
                                            remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache,
                                            desc="Running tokenizer on validation dataset",)
    
    """-------------------------------------------------------------------------"""
    #20. Prediction setting
    """-------------------------------------------------------------------------"""
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers,
                                                remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache,
                                                desc="Running tokenizer on prediction dataset",)

    """-------------------------------------------------------------------------"""
    #21. Data collator
    """-------------------------------------------------------------------------"""
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id,
                                            pad_to_multiple_of=8 if training_args.fp16 else None,)

    """-------------------------------------------------------------------------"""
    #22. Define metric
    """-------------------------------------------------------------------------"""
    metric = load_metric("rouge")

    """-------------------------------------------------------------------------------------------------------------
        TODO
    -------------------------------------------------------------------------------------------------------------"""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        #print("1.result: {}\n".format(result))
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        #print("2.result: {}\n".format(result))
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        #print("3.result: {}\n".format(result))

        return result

    """-------------------------------------------------------------------------"""
    #23. Initialize our Trainer
    """-------------------------------------------------------------------------"""
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None,
                            eval_dataset=eval_dataset if training_args.do_eval else None,
                            tokenizer=tokenizer, data_collator=data_collator,
                            compute_metrics=compute_metrics if training_args.predict_with_generate else None,)

    """-------------------------------------------------------------------------"""
    #24. Training
    """-------------------------------------------------------------------------"""
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    """-------------------------------------------------------------------------"""
    #25. Evaluation and prediction setting
    """-------------------------------------------------------------------------"""
    results = {}
    max_length = (training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length)

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    """-------------------------------------------------------------------------"""
    #26. Evaluation
    """-------------------------------------------------------------------------"""
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    """-------------------------------------------------------------------------"""
    #27. Prediction
    """-------------------------------------------------------------------------"""
    
    if training_args.do_predict:
        pred_list, ref_list = [], []
        df = pd.DataFrame(columns = ['reference' , 'system'])
        
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams)
        
        #print("predict_results: {}\n".format(predict_results))
        
        metrics = predict_results.metrics
        print("metrics: {}\n".format(metrics))

        max_predict_samples = (data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset))
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        print("metrics[predict_samples]: {}\n".format(metrics["predict_samples"]))        
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:

                predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                predictions = [pred.strip() for pred in predictions]
                print("predictions: {}\n".format(predictions))
                pred_list.append(predictions)

                references = predict_results.label_ids
                references = np.where(references != -100, references, tokenizer.pad_token_id)                
                references = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                references = [ref.strip() for ref in references]
                print("references: {}\n".format(references))

                ref_list.append(references)
                
        output_file = os.path.join(training_args.output_dir, "summaries.csv")
        df['reference'] = ref_list
        df['system'] = pred_list
        df.to_csv(output_file, index=False)
        #with open(output_prediction_file, "a+") as writer:
        #    writer.write("\n".join(predictions))

    return results

"""-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------"""
if __name__ == "__main__":
    main()