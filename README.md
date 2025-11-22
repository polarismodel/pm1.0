# Polaris LLM (PM1)
![Python](https://img.shields.io/badge/Python-3.10%2B-111111?style=for-the-badge&logo=python&logoColor=00F0FF)

![Code Style](https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge)

![License](https://img.shields.io/badge/License-Apache%202.0-111111?style=for-the-badge&color=00F0FF)

![Status](https://img.shields.io/badge/Status-TRAINING-00F0FF?style=for-the-badge&labelColor=111111)

This repository contains code to fine-tune GPT-2 into the Polaris LLM (version PM1).

Where outputs are saved
- Final model and tokenizer: `models/polaris-pm1/`

How to run training locally
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training (this script downloads datasets from Hugging Face):

```bash
python training/train_gpt2.py
```

GitHub Actions
- The workflow `.github/workflows/train-llm.yml` runs training and uploads the `models/polaris-pm1/` directory as an artifact named `polaris-pm1`.
