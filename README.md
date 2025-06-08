# PsychologicalPersuasion

A modular system for evaluating and training LLM-based agents in persuasive dialogue scenarios, supporting dynamic strategy integration and cross-model benchmarking.

## 📌 Overview

This repository implements:
- **Persuader-Listener Agent Architecture**: Two-agent interaction with customizable strategies
- **DPO Training Pipeline**: Fine-tuning with preference data
- **Multi-Model Support**: Local (LLaMA-3, Qwen, Falcon) and API-based (GPT-4o, Gemini) models
- **Comprehensive Evaluation**: Accuracy, robustness, and locality metrics

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KalinaEine/PsychologicalPersuasion.git
   cd PsychologicalPersuasion
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Configuration Setup
   
First, edit `config.yaml` with your specific settings.

### 2. Run Evaluation
   
#### Phase 1: Baseline Testing (No Strategy)
<pre><code>```bash python strategy_test.py \ --config_path config.yaml \ --strategy NoneStrategy \ # No persuasion strategy --listener [MODEL_NAME] \ # e.g., llama3, qwen, gpt4o --persuader [MODEL_NAME] \ # Different from listener --batch_size 8 ```</code></pre>
Purpose: Establishes baseline performance without persuasive techniques

#### Phase 2: Strategy Evaluation
```bash
python strategy_test.py \\
  --config_path config.yaml \\
  --strategy [STRATEGY_NAME] \\  # e.g., authority_effect
  --listener [MODEL_NAME] \\     # Same model for both
  --persuader [MODEL_NAME] \\    # Same as listener
  --batch_size 8
```
Available Strategies (defined in strategy_agent.py):

authority_effect - Leverage perceived expertise

flattery_trap - Excessive praise technique

repetition_effect - Message reinforcement

information_isolation - Controlled information flow

... [11+ others]

3. Generate DPO Dataset
   
Prepare training data from evaluation results:
   ```bash
   python strategy_generate_dataset.py --data_dir ./results --output_path ./dpo_data.jsonl
   ```

4. Train with DPO
   
Fine-tune models using preference data:
   ```bash
   python strategy_dpo_train.py
   ```

