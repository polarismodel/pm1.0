# training/train_gpt2.py

from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

MODEL_NAME = 'gpt2'




def main():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Load LMSYS conversational dataset
    ds1 = load_dataset("ytz20/LMSYS-Chat-GPT-5-Chat-Response")

    # Load code datasets
    code_datasets_info = [
        ("flytech/python-codes-25k", "code"),
        ("hsultanbey/javascript", "code"),
        ("goendalf666/sql-chat-instructions", "instruction"),
        ("supergoose/buzz_sources_094_cplusplus", "code"),
        ("spignelon/bash_history", "text"),
        ("SaeedRahmani/codeparrot_github_code_powershell", "code"),
        ("smcleod/golang-coder", "code"),
        ("nickrosh/Evol-Instruct-Code-80k-v1", "text"),
        ("mhhmm/typescript-instruct-20k", "code"),
        ("Neloy262/rust_instruction_dataset", "instruction"),
        ("marin-community/ar5iv-no-problem-markdown", "markdown"),
    ]

    def tokenize_lmsys(examples):
        texts = []
        for item in examples['content']:
            if isinstance(item, list):
                joined = ' '.join([d.get('content', '') for d in item if isinstance(d, dict)])
                texts.append(joined)
            else:
                texts.append(str(item))
        tokens = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    def make_tokenizer_fn(field):
        def fn(examples):
            texts = examples[field] if field in examples else [""] * len(examples[next(iter(examples))])
            tokens = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
            tokens['labels'] = tokens['input_ids'].copy()
            return tokens
        return fn

    # Tokenize LMSYS dataset
    tokenized_ds1 = {}
    for split in ds1:
        tokenized_ds1[split] = ds1[split].map(tokenize_lmsys, batched=True, num_proc=120, batch_size=256)

    # Load and tokenize code datasets
    from datasets import concatenate_datasets
    code_train_datasets = []
    for dataset_name, field in code_datasets_info:
        try:
            ds = load_dataset(dataset_name)
        except Exception as e:
            print(f"Warning: failed to load dataset {dataset_name}: {e}\nSkipping this dataset.")
            continue

        # Auto-detect the field to tokenize if the configured one isn't present
        candidate_fields = [field, 'content', 'code', 'text', 'instruction', 'markdown', 'source', 'snippet', 'example', 'complex_sentence']
        # inspect available columns in the first split that exists
        available_splits = [s for s in ds.keys()]
        chosen_field = None
        for split_name in available_splits:
            columns = ds[split_name].column_names
            for cand in candidate_fields:
                if cand in columns:
                    chosen_field = cand
                    break
            if chosen_field:
                break

        if not chosen_field:
            print(f"Warning: no suitable text/code field found for {dataset_name} (available columns: {available_splits}). Skipping.")
            continue

        tokenizer_fn = make_tokenizer_fn(chosen_field)
        for split in ds:
            if split == 'train':
                try:
                    tokenized = ds[split].map(tokenizer_fn, batched=True, num_proc=120, batch_size=256)
                    code_train_datasets.append(tokenized)
                except Exception as e:
                    print(f"Warning: failed to tokenize train split for {dataset_name}: {e}\nSkipping this split.")
                    continue

    # Combine all train splits
    train_datasets = []
    if 'train' in tokenized_ds1:
        train_datasets.append(tokenized_ds1['train'])
    train_datasets.extend(code_train_datasets)
    if train_datasets:
        combined_train = concatenate_datasets(train_datasets)
    else:
        raise ValueError("No train split found in any dataset.")

    training_args = TrainingArguments(
        output_dir='./models/polaris-pm1',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train,
    )

    trainer.train()
    # Save final model and tokenizer to the Polaris PM1 model directory
    model.save_pretrained('./models/polaris-pm1')
    tokenizer.save_pretrained('./models/polaris-pm1')

if __name__ == '__main__':
    main()
