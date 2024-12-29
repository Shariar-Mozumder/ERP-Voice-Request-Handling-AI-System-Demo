

import torch
from transformers import BertTokenizerFast,BertForTokenClassification
import numpy as np


tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
model = BertForTokenClassification.from_pretrained("./results/checkpoint-100")  


slot_label_map = {
    0: "O", 1: "B-project_id", 2: "I-project_id", 3: "B-reason", 4: "I-reason",
    5: "B-amount", 6: "I-amount", 7: "B-project_name", 8: "I-project_name",
    9: "B-status", 10: "I-status",11: "B-riyals", 12: "I-riyals" 
}



def decode_slots(tokens, predictions, slot_label_map):
    slots = {}
    current_slot = None
    current_value = []

    for token, pred_id in zip(tokens, predictions):
        label = slot_label_map[pred_id]

        # Handle B- and I- slots
        if label.startswith("B-"):  # Beginning of a new slot
            if current_slot:
                
                slots[current_slot] = tokenizer.convert_tokens_to_string(current_value)
            current_slot = label[2:]  # Extract slot name
            current_value = [token]  # Start a new slot
        elif label.startswith("I-") and current_slot == label[2:]:  # Continuation of the current slot
            current_value.append(token)
        else:  # No slot or "O"
            if current_slot:
                
                slots[current_slot] = tokenizer.convert_tokens_to_string(current_value)
                current_slot = None
                current_value = []

    if current_slot: 
        slots[current_slot] = tokenizer.convert_tokens_to_string(current_value)

    return slots


def predict_intent_and_slots(text, model, tokenizer, slot_label_map):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,  # Same as during training
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    predictions = predictions[:len(tokens)]  

   
    slots = decode_slots(tokens, predictions, slot_label_map)

    
    intent = "mock_intent" 

    return {"utterance": text, "slots": slots}

def get_slots(text):
    result = predict_intent_and_slots(text, model, tokenizer, slot_label_map)
    slots=result['slots']
    return slots

# Test the model
test_text = "Hey, I need to request money for a project name Abha University and id is 123 and the amount is 500 riyals"
result = predict_intent_and_slots(test_text, model, tokenizer, slot_label_map)

print("Prediction Result:")
print(result)
