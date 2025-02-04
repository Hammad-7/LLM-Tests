CONFIG = {
    "model_name": "unsloth/Qwen2.5-14B-Instruct",
    "max_seq_length": 2048,
    "dtype": None,  # Auto-detection
    "load_in_4bit": False,
    "lora": {
        "r": 16,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    },
    "training": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 5,
        "learning_rate": 2e-4,
        "fp16": True,
        "bf16": False,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "outputs",
        "report_to": "none",
    },
    "huggingface": {
        "repo_id": "hammad-007/Qwen2.5-14B-instruct",
    }
}