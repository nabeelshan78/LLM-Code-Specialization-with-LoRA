import torch
from torch.utils.data import Dataset
import evaluate
from IPython.display import display, Markdown


def format_prompt(example: dict, include_response: bool = True) -> list:
    """Formats a single data example into a prompt string."""
    # This function should handle both cases (with and without input)
    
    
    template_with_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    template_without_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    if example.get("input") and len(example["input"]) > 0:
        text = template_with_input.format(instruction=example["instruction"], input=example["input"])
    else:
        text = template_without_input.format(instruction=example["instruction"])

    if include_response:
        text += f"{example['output']}</s>"
    
    return [text]

def print_evaluation(instructions, expected_outputs, generated_outputs):
    """Visually compares generated outputs with expected outputs."""
    
    for i in range(len(generated_outputs)):
        
        output_md = (
            f"### Instruction {i+1}\n```\n{instructions[i]}\n```\n"
            f"### Expected Response {i+1}\n```\n{expected_outputs[i]}\n```\n"
            f"### Generated Response {i+1}\n```\n{generated_outputs[i]}\n```\n"
        )
        display(Markdown(output_md))
        display(Markdown("---"))

        
        

class ListDataset(Dataset):
    """Custom PyTorch Dataset to wrap a list of strings."""
    def __init__(self, original_list):
        self.original_list = original_list
    
    def __len__(self):
        return len(self.original_list)
    
    def __getitem__(self, i):
        return self.original_list[i]