from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import DatasetDict, Dataset
import json

# Load and preprocess dataset
def preprocess_data1(json_path, tokenizer):
    with open(json_path, "r") as f:
        data = json.load(f)["data"]

    tokenized_data = {"input_ids": [], "attention_mask": [], "labels": []}
    slot_label_map = {"O": 0}  # Start with "O" for outside slots
    label_id = 1

    for intent_data in data:
        for utterance in intent_data["utterances"]:
            text = utterance["text"]
            encoding = tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=128,  # Adjust based on your requirement
                return_offsets_mapping=True
            )
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            # Create slot labels for the tokens
            slot_labels = ["O"] * len(tokens)
            for slot, value in utterance["slots"].items():
                if value != "not specified":  # Skip unspecified slots
                    slot_tokens = tokenizer.tokenize(value)
                    for i in range(len(tokens) - len(slot_tokens) + 1):
                        if tokens[i:i + len(slot_tokens)] == slot_tokens:
                            slot_labels[i] = f"B-{slot}"
                            for j in range(1, len(slot_tokens)):
                                slot_labels[i + j] = f"I-{slot}"

            # Map slot labels to IDs
            for label in slot_labels:
                if label not in slot_label_map:
                    slot_label_map[label] = label_id
                    label_id += 1

            label_ids = [slot_label_map[label] for label in slot_labels]

            # Add to tokenized data
            tokenized_data["input_ids"].append(encoding["input_ids"])
            tokenized_data["attention_mask"].append(encoding["attention_mask"])
            tokenized_data["labels"].append(label_ids)

    # Print the slot_label_map to check for any incorrect labels
    print("Slot Label Map:", slot_label_map)
    
    # Convert to Dataset format
    dataset = Dataset.from_dict(tokenized_data)
    return DatasetDict({"train": dataset, "validation": dataset}), slot_label_map


# Update training preprocessing to handle multi-token slots like 'amount'
def preprocess_data(json_path, tokenizer):
    with open(json_path, "r") as f:
        data = json.load(f)["data"]

    tokenized_data = {"input_ids": [], "attention_mask": [], "labels": []}
    slot_label_map = {"O": 0}  # Start with "O" for outside slots
    label_id = 1

    for intent_data in data:
        for utterance in intent_data["utterances"]:
            text = utterance["text"]
            encoding = tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=128, 
                return_offsets_mapping=True
            )
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            # Create slot labels for the tokens
            slot_labels = ["O"] * len(tokens)
            for slot, value in utterance["slots"].items():
                if value != "not specified":  # Skip unspecified slots
                    # Tokenize the value (e.g., '500 dollars')
                    slot_tokens = tokenizer.tokenize(value)
                    for i in range(len(tokens) - len(slot_tokens) + 1):
                        if tokens[i:i + len(slot_tokens)] == slot_tokens:
                            slot_labels[i] = f"B-{slot}"
                            for j in range(1, len(slot_tokens)):
                                slot_labels[i + j] = f"I-{slot}"

            # Map slot labels to IDs
            for label in slot_labels:
                if label not in slot_label_map:
                    slot_label_map[label] = label_id
                    label_id += 1

            label_ids = [slot_label_map[label] for label in slot_labels]

            # Add to tokenized data
            tokenized_data["input_ids"].append(encoding["input_ids"])
            tokenized_data["attention_mask"].append(encoding["attention_mask"])
            tokenized_data["labels"].append(label_ids)

    # Convert to Dataset format
    dataset = Dataset.from_dict(tokenized_data)
    return DatasetDict({"train": dataset, "validation": dataset}), slot_label_map


# Load pre-trained multilingual model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# Preprocess dataset
json_path = "nlu_dataset.json"  # Replace with the actual JSON path
dataset, slot_label_map = preprocess_data(json_path, tokenizer)

# Load the model with the correct number of labels
model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", 
    num_labels=len(slot_label_map)  # Set num_labels based on the size of the slot_label_map
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Fine-tuning logic
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

trainer.train()
