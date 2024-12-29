import json
from whisper_stt import transcribe_audio_raw
from intent_recognition import get_intent_and_amount
from generate_response import generate_voice_response
from test_NLU import get_slots

DATABASE_PATH = "database.json"

def load_database():
    with open(DATABASE_PATH, "r") as db_file:
        return json.load(db_file)

def save_to_database(data):
    with open(DATABASE_PATH, "w") as db_file:
        json.dump(data, db_file, indent=4)

def handle_request(audio_file):
    
    text = transcribe_audio_raw(audio_file)
    
    
    intent_data = get_intent_and_amount(text)
    intent=intent_data.get('intent')
    intent=intent.replace("_", " ").title()
    amount_data=intent_data.get('amount_data')
    amount=amount_data.get('amount')
    currency=amount_data.get('currency')
    slots=get_slots(text)
    slots['amount']=amount+' '+currency
    if intent is not None:
        response=f"You have requested for the task: Intent: {intent}, Project: {slots.get('project_name')}. Project ID: {slots.get('project_id')}.  Amount: {slots.get('amount')}. Task ID: {slots.get('task_id')} and Status: {slots.get('status')}. Please Confirm by typing your response: Yes or No: "
        generate_voice_response(response) 
        user_input=input("Please type your response: Yes or No: ")
    
        if user_input.lower()=="yes":
            # Prepare the data to save
            request_data = {
                "project": slots.get("project_name"),
                "project_id": slots.get("project_id"),
                "amount": amount,
                "Intent": intent,
                "task_id": slots.get("task_id"),
                "status": slots.get("status"),
            }

           
            database = load_database()
            database["requests"].append(request_data)
            save_to_database(database)
            generate_voice_response("Thank you for your response, Your request has been confirmed successfully.")
            
        elif user_input.lower()=="no":
            generate_voice_response("Thank you for your response, You have denied the confirmation request.")
        else:
            generate_voice_response("You have typed an invalid response.")
    else:
        response = "Sorry, I did not understand your request."
        generate_voice_response(response)
    return response
    

if __name__ == "__main__":
    
    user_audio = "input_audio.wav"
    # audio_file = open(user_audio, "rb")
    
    print(handle_request(user_audio))
