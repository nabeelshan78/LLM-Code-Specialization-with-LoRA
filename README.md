# Specializing LLMs for Code Generation via Parameter-Efficient Fine-Tuning (LoRA)

### Project Goal: Transforming a general-purpose language model into a specialized, instruction-following code generator with minimal computational cost.

This project demonstrates my ability to adapt and specialize foundation models for complex, domain-specific tasks. I successfully fine-tuned the `facebook/opt-350m` model on the `CodeAlpaca-20k` dataset, turning its generic, often irrelevant outputs into precise and accurate code snippets based on user instructions.

The key achievement was leveraging **Low-Rank Adaptation (LoRA)**, a Parameter-Efficient Fine-Tuning (PEFT) technique. This allowed me to achieve high-performance specialization by training **less than 1%** of the model's total parameters, showcasing an efficient and scalable approach to model customization.

---

## 🚀 Key Result: Before vs. After LoRA Fine-Tuning

The impact of instruction fine-tuning is best shown by comparing the model's performance on the same prompt before and after the LoRA process. The transformation from an incoherent text generator to a functional code generator is stark.

| | Before Fine-Tuning (Base `opt-350m`) | After LoRA Fine-Tuning (Specialized Model) |
| :--- | :--- | :--- |
| **Instruction** | `Create a javascript class that takes in two arguments and prints them out when called.` | `Create a javascript class that takes in two arguments and prints them out when called.` |
| **Model Output** | `Once you have a response, write it down. Write it down. Write it down... ### Instructions: Add a JavaScript object to the end of the class.` | `class PrintNumber{ constructor(num1, num2){ this.num1 = num1; this.num2 = num2; } printNumber(){ console.log(\`${this.num1}, ${this.num2}\`); } }` |
| **Assessment** | ❌ **Failure:** The base model completely fails to understand the instruction, generating repetitive, nonsensical text. | ✅ **Success:** The fine-tuned model correctly interprets the instruction and generates a perfectly valid and functional JavaScript class. |

### Training Performance

The model demonstrated successful learning, as evidenced by the consistently decreasing training loss curve. This indicates effective convergence on the task of following coding instructions.

![Training Loss Curve](train_loss.png)

---

## 🛠️ Technical Methodology

My approach involved a systematic workflow, from data strategy to model training and evaluation, leveraging the state-of-the-art Hugging Face ecosystem.

#### 1. Data Strategy & Prompt Engineering
-   **Dataset:** Leveraged the `CodeAlpaca-20k` dataset, which contains a rich set of instruction-output pairs tailored for code generation tasks.
-   **Prompt Templating:** I designed a structured prompt template to format the dataset. This crucial step creates a consistent input format (`### Instruction: ... ### Response: ...`) that enables the model to learn the instruction-following pattern effectively. This strategy is key to successful instruction fine-tuning.

#### 2. Parameter-Efficient Fine-Tuning (PEFT) with LoRA
-   **Core Technique:** Implemented Low-Rank Adaptation (LoRA) to inject small, trainable matrices into the attention layers (`q_proj`, `v_proj`) of the transformer architecture.
-   **Efficiency:** This approach is highly efficient. Instead of training all **350M** parameters of the base model, I only updated the LoRA adapters, which constituted **~1.5M** parameters. This represents a **>99% reduction in trainable parameters**, drastically lowering VRAM requirements and training time.

#### 3. Optimized Training with TRL
-   **Trainer:** Utilized the `SFTTrainer` from the Hugging Face `trl` (Transformer Reinforcement Learning) library, which is specifically optimized for supervised fine-tuning on instruction-based datasets.
-   **Targeted Loss Calculation:** Employed `DataCollatorForCompletionOnlyLM` to ensure the loss function was computed *only* on the `### Response:` section of the prompts. This focuses the model's learning entirely on generating the correct output, rather than memorizing the instruction, leading to more efficient and effective training.

---

## 📂 Repository Structure
```bash
LLM-Code-Specialization-with-LoRA/
├── LoRA_Finetuning_Walkthrough.ipynb        # Main Jupyter Notebook: full workflow of LoRA fine-tuning.
├── config.py                                # LoRA + SFTTrainer configuration parameters.
├── utils.py                                 # Helper functions for prompt formatting and evaluation.
├── train_loss.png                           # Training loss curve visualization.
├── instruction-tuning-log-history-lora.json # Training logs for reproducibility.
├── generated_outputs_base.pkl               # Saved outputs from the base model (pre-tuning).
├── instruction-tuning-generated-outputs-base.pkl  # Evaluation results: base model.
├── instruction-tuning-generated-outputs-lora.pkl  # Evaluation results: LoRA fine-tuned model.
└── README.md                                # Project documentation (you are here!).
```

---

## Reproduce My Results

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nabeelshan78/LLM-Code-Specialization-with-LoRA.git
    cd LLM-Code-Specialization-with-LoRA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    # Or install manually: pip install torch transformers datasets peft trl
    ```

3.  **Launch the notebook:**
    Open and run the cells in `LoRA_Finetuning_Walkthrough.ipynb` using Jupyter Lab or a similar environment.

---

## 💻 Technology Stack & Core Competencies

This project showcases my proficiency with the modern AI/ML development stack and key deep learning concepts.

-   **Languages & Frameworks:** Python, PyTorch
-   **Hugging Face Ecosystem:** `transformers`, `peft`, `trl`, `datasets`
-   **Core Concepts:**
    -   Large Language Models (LLMs)
    -   Instruction Fine-Tuning
    -   Parameter-Efficient Fine-Tuning (PEFT)
    -   Low-Rank Adaptation (LoRA)
    -   Prompt Engineering
    -   Model Evaluation & Performance Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ecosystem-yellow.svg?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-f37626.svg?style=for-the-badge&logo=jupyter)
