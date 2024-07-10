import logging
import random
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

# Ajuste do max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer
)
from peft import LoraConfig, get_peft_model
from trainer_4bit import DPOTrainer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

logger = logging.getLogger(__name__)

def convert_model_to_dtype(model, dtype):
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.do_eval = False
    training_args.per_device_train_batch_size = 1  # Reduzir tamanho do lote
    training_args.gradient_accumulation_steps = 8  # Aumentar acumulação de gradientes para compensar o menor lote
    training_args.gradient_checkpointing = True  # Habilitar gradient checkpointing
    training_args.fp16 = False  # Desativar fp16
    training_args.bf16 = True  # Ativar bf16 para corresponder ao DeepSpeed

    torch.cuda.empty_cache()  # Liberar cache antes de iniciar o treinamento
    torch.cuda.reset_peak_memory_stats()  # Resetar as estatísticas de memória de pico

    num_iteration = 1
    for i in range(num_iteration):
        model = main_inner(model_args, data_args, training_args)
        logger.info(f"-------------------------Finished Iteration {i+1}---------------------------------")
        # Log dos detalhes do modelo
        for name, param in model.named_parameters():
            logger.info(f"Parameter {name}: dtype={param.dtype}, shape={param.shape}")

def main_inner(model_args, data_args, training_args):
    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)

    raw_datasets = get_datasets(data_args, splits=["train"])
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names = [x for x in column_names if x not in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']]

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo", "skip_system_message": True},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=True,
        **model_kwargs,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    ref_model = model if not model_args.use_peft else None

    # Convert model to bfloat16 if necessary
    if training_args.bf16:
        model = convert_model_to_dtype(model, torch.bfloat16)
        ref_model = convert_model_to_dtype(ref_model, torch.bfloat16)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    checkpoint = None
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    trainer.accelerator.wait_for_everyone()
    logger.info("*** Training complete! ***")

if __name__ == "__main__":
    main()