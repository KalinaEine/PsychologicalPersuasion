# Psychological Persuasion

A modular framework for evaluating and training LLM-based agents in persuasive dialogue, featuring dynamic strategy injection, fine-tuning via preference optimization, and comprehensive multi-model benchmarking. Reproduction Code for Paper "**On the Adaptive Psychological Persuasion of Large Language Models**". The preprint of our paper is publicly available at [this link]().

## 📌 Overview

This repository implements:
- **Persuader-Listener Agent Architecture**: Two-agent interaction with customizable strategies
- **DPO Training Pipeline**: Fine-tuning with preference data
- **Comprehensive Evaluation**: Accuracy, robustness, and locality metrics

## 🛠️ Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/KalinaEine/PsychologicalPersuasion.git
   cd PsychologicalPersuasion
   ```
   
### 2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Acquire Dataset

We use the following files from the Counterfact-1K dataset:

`counterfact-val.json`: used for model evaluation

`counterfact-train.json`: used to generate DPO training data

You can access the dataset here: [Counterfact-1K](https://github.com/Jometeorie/KnowledgeSpread/tree/main/data/counterfact)

## 🚀 Quick Start

### 1. Configuration Setup
   
First, edit `config.yaml` with your specific settings.

### 2. Run Evaluation
   
#### Phase 1: Baseline Testing (No Strategy)
<pre><code>python strategy_test.py \
  --config_path config.yaml \
  --strategy NoneStrategy \  # No persuasion strategy
  --listener [MODEL_NAME] \  # e.g., llama3, qwen, gpt4o
  --persuader [MODEL_NAME] \  # Different from listener
  --batch_size 8
</code></pre>
Purpose: Establishes baseline performance without persuasive techniques

#### Phase 2: Strategy Evaluation
<pre><code>python strategy_test.py \
  --config_path config.yaml \
  --strategy [STRATEGY_NAME] \  # e.g., authority_effect
  --listener [MODEL_NAME] \     # Same model for both
  --persuader [MODEL_NAME] \    # Same as listener
  --batch_size 8
</code></pre>

Available Strategies (defined in `strategy_agent.py`):

**authority_effect** - Leverage perceived expertise

**flattery_trap** - Excessive praise technique

**repetition_effect** - Message reinforcement

**information_isolation** - Controlled information flow

... [7+ others]

#### Phase 3: Four Semantic Domains Evaluation

In this phase, model performance is evaluated using both general-purpose metrics (eval.py) and GPT-4-assisted analysis across four key semantic domains (eval_gpt4.py).

#### Basic Metrics Evaluation (`eval.py`)

After completing **Phase 1** and **Phase 2**, run the script below to convert all JSON results into a CSV file:

```bash
python eval.py
```

#### GPT-4 Categorized Evaluation (`eval_gpt4.py`)

Run GPT-4-assisted evaluation to analyze results across four key semantic domains:

```bash
python eval_gpt4.py
```

| Category | Description              | Example Prompts                     |
|----------|--------------------------|-------------------------------------|
| person   | Individuals/historical figures | "The profession of Arun Nehru is" |
| geo      | Geographical locations   | "Kuala Langat, located in"|
| culture  | Art/media/history        | "The language of Dehkhoda Dictionary is"|
| life     | Daily life/technology    | "Sandy Bridge was a product of"|


### 3. Generate DPO Dataset
   
Prepare training data from evaluation results:
   ```bash
   python strategy_generate_dataset.py
   ```

### 4. Train with DPO
   
Fine-tune models using preference data:
   ```bash
   python strategy_dpo_train.py
   ```

Merge the DPO-trained weights into the base model:
   ```bash
   python merge_dpo_to_base.py
   ```

### 5. Model Testing Pipeline

#### Test Fine-tuned DPO Models

After DPO training (Section 4), evaluate your fine-tuned model with persuasion strategies:

<pre><code># Original vs DPO comparison
   python strategy_test.py \
  --config_path config.yaml \
  --strategy [STRATEGY_NAME] \  # e.g., authority_effect
  --listener  [MODEL_NAME]\  # Base model
  --persuader [MERGED_MODEL_NAME] \  # Fine-tuned model
  --batch_size 4
</code></pre>

#### MMLU Benchmark Evaluation

Assess general knowledge capabilities:

```bash
python MMLU.py
```
