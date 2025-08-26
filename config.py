from peft import LoraConfig, TaskType
from trl import SFTConfig

MODEL_PATH = "./facebook-opt-350m"

def get_lora_config():
    """Returns the LoRA configuration."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def get_training_args():
    """Returns the SFT training arguments."""
    return SFTConfig(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=25,
        save_strategy="epoch",
        max_seq_length=1024,
        fp16=True,
    )



def get_generation_params():
    """Returns a dictionary of parameters for the text-generation pipeline."""
    return {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False
    }