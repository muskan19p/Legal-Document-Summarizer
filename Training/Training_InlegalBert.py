import os
import json
import torch
import numpy as np
from datasets import Dataset
import evaluate
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)

# === Config === #
BASE_MODEL_NAME = "law-ai/InLegalBERT"
SUMM_MODEL_NAME = "t5-small"
HF_REPO_NAME = "RvShivam/BhLegalBert"
DATA_DIR = "Dataset/Data_BNS_BNSS_BSA"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# === Utility Functions === #
def load_statute_texts(filenames):
    texts = []
    for fname in filenames:
        with open(fname, "r", encoding="utf-8") as f:
            for item in json.load(f):
                text = item.get("subsection_text") or item.get("section_text")
                if text: texts.append(text)
    return texts

def load_mapping_data(filenames):
    inputs, targets = [], []
    for fname in filenames:
        with open(fname, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if "bnss_section" in entry:
                    input_text = f"BNSS Section {entry['bnss_section']}: {entry['bnss_subject']} | CrPC Section {entry['crpc_section']}: {entry['crpc_subject']}"
                elif "bns_section" in entry:
                    input_text = f"BNS Section {entry['bns_section']}: {entry['bns_subject']} | IPC Section {entry['ipc_section']}: {entry['ipc_subject']}"
                elif "bsa_section" in entry:
                    input_text = f"BSA Section {entry['bsa_section']}: {entry['bsa_subject']} | IEA Section {entry['iea_section']}: {entry['iea_subject']}"
                else:
                    continue
                summary = entry.get("summary", "")
                if input_text and summary:
                    inputs.append(input_text)
                    targets.append(summary)
    return inputs, targets

def load_case_law_data(filenames):
    cases = []
    for fname in filenames:
        with open(fname, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                contexts = [m.get("context", "") for m in entry.get("statute_mentions", []) if "context" in m]
                if contexts:
                    combined = f"{entry.get('case_title', '')} | " + " ".join(contexts)
                    cases.append(combined)
    return cases

# === Metrics === #
rouge = evaluate.load("rouge")
def compute_rouge(eval_pred):
    predictions, labels = eval_pred

    # Handle tuple output from model (e.g., (logits,))
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # If predictions are logits (floats), we need to argmax to get token ids
    if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Ensure both are lists of token ids
    predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels

    labels = [
    [(token if token >= 0 else tokenizer.pad_token_id) for token in label]
    for label in labels
    ]


    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return {
        key: round(value * 100, 4) for key, value in rouge_scores.items()
    }

def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# === 1. MLM Training === #
print("\n🔡 Loading MLM data...")
statute_files = [f"{DATA_DIR}/Statutes/BNSS_sections.json", f"{DATA_DIR}/Statutes/BNS_sections.json", f"{DATA_DIR}/Statutes/BSA_sections.json"]
statute_texts = load_statute_texts(statute_files)
mlm_dataset = Dataset.from_dict({"text": statute_texts}).train_test_split(test_size=0.2)

def tokenize_mlm(batch): return tokenizer(batch["text"], truncation=True, max_length=512)
tokenized_mlm = mlm_dataset.map(tokenize_mlm, batched=True)
mlm_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)

mlm_trainer = Trainer(
    model=mlm_model,
    args=TrainingArguments(
    output_dir="./mlm",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    fp16=True,
    logging_dir="./logs/mlm",         # ✅ Added
    logging_steps=50,                 # ✅ Added
    push_to_hub=True,
    hub_model_id=HF_REPO_NAME
),
    train_dataset=tokenized_mlm["train"],
    eval_dataset=tokenized_mlm["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
    tokenizer=tokenizer
)

print("🚀 Starting MLM training...")
mlm_trainer.train()
mlm_trainer.push_to_hub()

# === 2. Summarization Training === #
print("\n📘 Loading summarization (mapping) data...")
mapping_files = [f"{DATA_DIR}/Comparative/CrPC_BNSS_mappings.json", f"{DATA_DIR}/Comparative/IPC_BNS_mappings.json", f"{DATA_DIR}/Comparative/BSA_IEA_mappings.json"]
inputs, targets = load_mapping_data(mapping_files)
summ_dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets}).train_test_split(test_size=0.2)

def tokenize_summarization(batch):
    inputs = tokenizer(batch["input_text"], max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], max_length=128, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_summ = summ_dataset.map(tokenize_summarization, batched=True)
summ_model = T5ForConditionalGeneration.from_pretrained(SUMM_MODEL_NAME)

summ_trainer = Trainer(
    model=summ_model,
    args=TrainingArguments(
        output_dir="./summarization",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        fp16=True,
        logging_dir="./logs/summ",         # ✅ Added
        logging_steps=50,                 # ✅ Added
        push_to_hub=True,
        hub_model_id=HF_REPO_NAME
    ),
    train_dataset=tokenized_summ["train"],
    eval_dataset=tokenized_summ["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=summ_model),
    compute_metrics=compute_rouge
)

print("🧠 Starting summarization fine-tuning...")
summ_trainer.train()
summ_trainer.push_to_hub()

# === 3. Retrieval Training === #
print("\n📂 Loading retrieval (case law) data...")
case_files = [f"{DATA_DIR}/Case_Files/BNSS_cases.json", f"{DATA_DIR}/Case_Files/BNS_cases.json", f"{DATA_DIR}/Case_Files/BSA_cases.json"]
case_texts = load_case_law_data(case_files)
retrieval_dataset = Dataset.from_dict({
    "text": case_texts,
    "label": [1] * len(case_texts)  # dummy binary label
}).train_test_split(test_size=0.2)

def tokenize_retrieval(batch): return tokenizer(batch["text"], truncation=True, max_length=512)
tokenized_ret = retrieval_dataset.map(tokenize_retrieval, batched=True)
retrieval_model = AutoModelForSequenceClassification.from_pretrained("./mlm", num_labels=2)

retrieval_trainer = Trainer(
    model=retrieval_model,
    args=TrainingArguments(
        output_dir="./retrieval",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        fp16=True,
        logging_dir="./logs/retrieval",         # ✅ Added  
        logging_steps=50,                 # ✅ Added
        push_to_hub=True,
        hub_model_id=HF_REPO_NAME
    ),
    train_dataset=tokenized_ret["train"],
    eval_dataset=tokenized_ret["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy
)

print("🔍 Starting retrieval fine-tuning...")
retrieval_trainer.train()
retrieval_trainer.push_to_hub()

# === Final Note === #
print("\n✅ All training complete. Models pushed to Hugging Face Hub.")
