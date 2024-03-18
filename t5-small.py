import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel, Value, Features
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import json
import evaluate

# Data Preparation

df = pd.read_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus_filtered.csv')
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'id'})
df['id'] = df['id'].astype(str)
df['translation'] = df.apply(lambda row: {"biased": row['biased'], "unbiased": row['unbiased']}, axis=1)
df = df.drop(['biased', 'unbiased'], axis=1)
df.to_csv('/Users/architg/Documents/GitHub/final-year-project/data/translation/data.csv', index=False)

dataset = load_dataset('csv', data_files = '/Users/architg/Documents/GitHub/final-year-project/data/translation/data.csv')
dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)

def convert_to_dict(example):
    count = 0
    if example is None:
        return None
    
    try:
        if "translation" in example:
            translation_str = example["translation"]
            if translation_str is not None:  # Check if translation_str is not None
                translation_dict = json.loads(translation_str.replace("'", "\""))
                # print(f"Translation dict: {translation_dict}")
                return {"translation": translation_dict}
            else:

                print("Translation string is None")
                print(example["translation"])
                return {"translation": None}  # Return dictionary with None translation for None translation strings
        else:

            print("Translation key is missing")
            print(example["translation"])
            return {"translation": None}  # Return dictionary with None translation if translation key is missing
    except json.JSONDecodeError as e:

        # print(f"Error decoding JSON: {e}")
        # print(f"Problematic translation string: {translation_str}")
        return {"translation": None}  # Return dictionary with None translation for problematic examples

# Apply the conversion function to the entire dataset
dataset = dataset.map(convert_to_dict)

# Remove problematic examples from the dataset
dataset = dataset.filter(lambda x: x['translation'] is not None)


checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "biased"
target_lang = "unbiased"
prefix = "translate biased to unbiased: "


def preprocess_function(examples):
    if examples is None or "translation" not in examples:
        return None
    
    inputs = []
    targets = []
    for example in examples["translation"]:
        if example is not None and source_lang in example and target_lang in example:
            input_text = prefix + example[source_lang]
            target_text = example[target_lang]
            inputs.append(input_text)
            targets.append(target_text)
            # print(input_text, target_text)

    if not inputs or not targets:
        return None
    
    # Tokenize inputs and targets with padding/truncation
    model_inputs = tokenizer(inputs, text_target=targets, max_length=1001, padding="max_length", truncation=True, return_overflowing_tokens=True)
    return model_inputs

tokenized_data = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()