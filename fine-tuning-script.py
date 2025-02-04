import os
import json
import pandas as pd
import emoji
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

# Load LoRA configuration from JSON file
def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

# Function to clean text
def clean_text(text):
    text = " ".join(text.split())
    return emoji.demojize(text)

# Function to format training prompts
def formatting_prompts_func(examples, prompt_template):
    texts = [prompt_template.format(c, i, o) for c, i, o in zip(examples["context"], examples["input"], examples["output"])]
    return {"text": texts}

# Main function to train and upload model
def train_model(config_path, data_path):
    config = load_config(config_path)
    model_name = config["model_name"]
    max_seq_length = config.get("max_seq_length", 2048)
    dtype = config.get("dtype", None)
    load_in_4bit = config.get("load_in_4bit", False)
    output_dir = config.get("output_dir", "outputs")
    hf_repo_name = config["hf_repo_name"]
    prompt_template = """Below is a conversation between a user and an assistant. The user asks a question or makes a statement, and the assistant responds. Your task is to continue the conversation by responding to the assistant. You can ask a question, make a statement, or take the conversation in a new direction. Your response should be in English and should make sense in the context of the conversation. Ensure that your response mirrors the assistant's speaking style as demonstrated in the examples.

    ### Context:
    {}

    ### Input:
    {}

    ### Response:
    {}
    """


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"].get("bias", "none"),
        use_gradient_checkpointing=config["lora"].get("use_gradient_checkpointing", "unsloth"),
        random_state=config["lora"].get("random_state", 3407),
        use_rslora=config["lora"].get("use_rslora", False),
        loftq_config=config["lora"].get("loftq_config", None),
    )

    df = pd.read_csv(data_path)
    df['input'] = df['input'].apply(clean_text)
    df['output'] = df['output'].apply(clean_text)
    df['context'] = df['context'].apply(clean_text)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: formatting_prompts_func(x, prompt_template), batched=True)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config["training"]["batch_size"],
            gradient_accumulation_steps=config["training"]["grad_accum_steps"],
            warmup_steps=config["training"]["warmup_steps"],
            num_train_epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    trainer.train()
    
    api = HfApi()
    api.set_access_token(os.getenv('HF_TOKEN'))
    model.push_to_hub(hf_repo_name, use_auth_token=True)
    tokenizer.push_to_hub(hf_repo_name, use_auth_token=True)

if __name__ == "__main__":
    train_model("config.py", "preprocessed_chat.csv")