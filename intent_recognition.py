
import re
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Any


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define the Intent Rules
dataset = {
    "data": [
        {
            "intent": "request_money",
            "utterances": [
                {"text": "I need to request money for project 223 to buy some tools, the amount I need is 500 riyals"},
                {"text": "Please add a money request for the project Abha University for 300 riyals"},
                {"text": "I need 1000 riyals for project 445 to purchase some equipment"},
                {"text": "Can you initiate a money request for project 678 with an amount of 250 riyals for team activities?"},
                {"text": "Requesting 800 riyals for the project Green Energy for office supplies"}
            ]
        },
        {
            "intent": "submit_task",
            "utterances": [
                {"text": "I have completed the task 1025, please mark it as done"},
                {"text": "Mark task 3054 as finished in the system"},
                {"text": "Task 8899 has been completed, update its status"},
                {"text": "Please mark task 1122 as done, I just finished it"},
                {"text": "Set the status of task 4500 to finished"}
            ]
        },
        {
            "intent": "get_project_status",
            "utterances": [
                {"text": "Can you tell me the status of project 223?"},
                {"text": "What is the current progress on project Abha University?"},
                {"text": "I need an update on project 445. What is its status?"},
                {"text": "Could you check and let me know the status of the Smart City project?"},
                {"text": "What’s the progress on the renewable energy project?"}
            ]
        }
    ]
}

def extract_amount_with_context(text: str) -> Dict[str, Any]:
    """Extract the amount (in currency) along with the currency term and context using regex."""
    
    # Adjust the regex to capture the amount and surrounding words
    match = re.search(r'(\d+)\s*(riyals?|reels?|rils?|reel?|dollars?|money|amount|usd|euro|pounds?)\s*(\w{1,20})?(\w{1,20})?', text.lower())
    
    if match:
        # Extract the amount and the currency type
        amount = match.group(1)
        currency = match.group(2)
        additional_info = f"{match.group(3)} {match.group(4)}".strip() if match.group(3) or match.group(4) else None
        return {"amount": amount, "currency": currency, "context": additional_info}
    return None

def get_intent_and_amount(text: str) -> Dict[str, Any]:
    """
    Extract intent and amount (if present) from a given text using a similarity model.
    """
    best_match = None
    best_score = 0
    intent = "unknown"
    amount_data = extract_amount_with_context(text)  

    # Now, let's detect the intent from the dataset
    for intent_data in dataset["data"]:
        for utterance in intent_data["utterances"]:
            # Compute similarity
            similarity_score = util.pytorch_cos_sim(
                model.encode(text, convert_to_tensor=True),
                model.encode(utterance["text"], convert_to_tensor=True)
            ).item()

            if similarity_score > best_score:
                best_score = similarity_score
                best_match = utterance
                intent = intent_data["intent"]

    return {"intent": intent, "amount_data": amount_data, "score": best_score}

# Example test
user_text = "Hey, I need to request money for a project name Abha University and id is 123 and the amount is 500 riyals"
result = get_intent_and_amount(user_text)
print(result)
