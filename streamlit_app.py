import json
import streamlit as st
from whisper_stt import transcribe_audio
from intent_recognition import get_intent_and_amount
from generate_response import generate_voice_response
from test_NLU import get_slots

DATABASE_PATH = "database.json"


def load_database():
    try:
        with open(DATABASE_PATH, "r") as db_file:
            return json.load(db_file)
    except FileNotFoundError:
        return {"requests": []}


def save_to_database(data):
    with open(DATABASE_PATH, "w") as db_file:
        json.dump(data, db_file, indent=4)


def handle_request(audio_file):
    while True:
        # Step 1: Transcribe audio to text
        text = transcribe_audio(audio_file)

        # Step 2: Process the text to extract intent and required slots
        intent_data = get_intent_and_amount(text)
        intent = intent_data.get("intent")
        if intent:
            intent = intent.replace("_", " ").title()
        amount_data = intent_data.get("amount_data")
        amount = amount_data.get("amount") if amount_data else None
        currency = amount_data.get("currency") if amount_data else ""
        slots = get_slots(text)
        project_name = slots.get("project_name")
        project_id = slots.get("project_id")
        task_id = slots.get("task_id")
        status = slots.get("status")

        # Ensure mandatory fields are present
        if not intent or not amount or not project_id:
            generate_voice_response(
                "Mandatory fields are missing. Please provide the required information again."
            )
            st.warning("Mandatory fields missing. Please try again.")
            continue

        # Display extracted data on Streamlit UI
        st.write("### Extracted Data")
        st.text(f"Extracted Text: {text}")
        st.text(f"Intent: {intent}")
        st.text(f"Project Name: {project_name}")
        st.text(f"Project ID: {project_id}")
        st.text(f"Amount: {amount} {currency}")
        st.text(f"Task ID: {task_id}")
        st.text(f"Status: {status}")

        response = (
            f"You have requested for the task: Intent: {intent}, "
            f"Project: {project_name}. Project ID: {project_id}. "
            f"Amount: {amount} {currency}. Task ID: {task_id} and Status: {status}. "
            "Please confirm by typing your response: Yes or No."
        )
        generate_voice_response(response)

        # User confirmation
        # user_input = st.text_input("Type your response (Yes/No):")
        user_input = st.text_input("Type 'yes' or 'no':").strip().lower()
        if user_input.lower() == "yes":
            request_data = {
                "project": project_name,
                "project_id": project_id,
                "amount": amount,
                "Intent": intent,
                "task_id": task_id,
                "status": status,
            }

            # Save to database
            database = load_database()
            database["requests"].append(request_data)
            save_to_database(database)

            generate_voice_response(
                "Thank you for your response, Your request has been confirmed successfully."
            )
            st.success("Request confirmed and saved successfully.")
            st.session_state.reset = True
            break
        elif user_input.lower() == "no":
            generate_voice_response(
                "Thank you for your response, You have denied the confirmation request."
            )
            st.warning("Request denied.")
            st.session_state.reset = True
            break
        # else:
        #     generate_voice_response("You have typed an invalid response.")
        #     st.error("Invalid response. Please try again.")
        #     continue


# Streamlit App
st.title("ERP Voice Request Handling AI System-Demo")
st.write("Upload an audio file and extract information from the request.")

# Upload audio file
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if audio_file:
    st.write("### Processing Audio Input")
    handle_request(audio_file)

# Display database records
st.write("### Saved Requests in Database")
database = load_database()
if database["requests"]:
    st.json(database)
else:
    st.write("No requests found.")
