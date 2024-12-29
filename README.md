# ERP Voice Request Handling AI System - Demo

This demo project implements an AI Agent that interacts with users through voice commands, handling tasks like form filling, database operations in an ERP system, requesting money, and more.

## Project Components

### 1. Speech-to-Text (STT)
- Converts user speech into text using the Whisper model.
- **Model Used**: `openai/whisper-tiny.en`  
- The model is open-source, lightweight, and supports multiple languages like Arabic.
- Implementation: `whisper_stt.py`

### 2. Natural Language Understanding (NLU)
- Uses a BERT-based multilingual model for intent recognition and slot filling.
- **Model Used**: `bert-base-multilingual-cased`  
- The model is lightweight, multilingual, and well-suited for this task.
- Implementation: `test_NLU.py`

### 3. Fine-Tuning NLU
- Custom JSON dataset (`nlu_dataset.json`) was created for fine-tuning the NLU model.
- Fine-tuning was crucial for better language and context understanding.
- Training was performed for 100 epochs, producing satisfactory results.
- Fine-tuned model stored in the `results/checkpoint-100/` folder.
- Implementation: `fine_tune_nlu.py`

### 4. Intent Recognition
- Fine-tuned NLU model struggled with intent and amount recognition due to limited data.
- **Solution**: Used `sentence-transformers/all-mpnet-base-v2` for similarity search and successfully retrieved intent and amount details.
- **Intent Rules**:
  The system supports specific intents for ERP tasks such as:
  - `request_money`
  - `submit_task`
  - `get_project_status`
- Intent recognition and amount extraction are implemented in `intent_recognition.py`.

### 5. Text-to-Speech (TTS)
- Provides user feedback via voice using `pyttsx3`.
- Essential for human-machine interaction and confirming actions.
- Implementation: `generate_response.py`

### 6. Main Application
- Merges all modules (STT, NLU, Intent Recognition, TTS) to execute project logic.
- Simple JSON-based database (`database.json`) is used for storing and retrieving requests.
- Mandatory fields and validations are included.
- Streamlit app created for demonstration.
- Implementation: `main.py` (logic) and `streamlit_app.py` (app).

---

## Project Setup

### 1. Clone the Project
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```
Note: During the demo, the app may show some minor errors in Streamlit. These do not impact the core functionality and can be ignored.


## Intent Examples
### Intent: `request_money`
-**Example Utterances**:
  -"I need to request money for project 223 to buy some tools, the amount I need is 500 riyals."
  -"Please add a money request for the project Abha University for 300 riyals."
