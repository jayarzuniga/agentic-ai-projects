from dotenv import load_dotenv
from google.cloud import firestore
from lanchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
import os

"""
1. Create a Firebase Account
2. Create a new firebase project
    - Copy the project ID
3. Create a firestore database in the firebase project
4. install the google cloud CLI in computer
-> gcloud init
-Authenticate the google cloud cli with your google account
-set you default project to the new firebase project you created.
5. Enable the firestore API in the Google Cloud Console:
"""

load_dotenv()

PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID") # Firebase Project ID
SESSION_ID = os.getenv("FIRESTORE_SESSION_ID") # Username or a Unique ID
COLLECTION_ID = os.getenv("FIRESTORE_COLLECTION_ID") # Collection ID ex. chat_history

print("initializing firestore client")
client = firestore.Client(project=PROJECT_ID)

print("initializing chat history")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection_id = COLLECTION_ID,
    client = client
)
print("initializing chat model")
print("Current Chat History: ", chat_history.messages)

model = ChatOpenAI(
    
)

print("Start chatting with the model. type quit to exit")

while True:
    human_input = input("User: ")
    if human_input.lower() == "quit":
        break
    
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(human_input)
    chat_history.add_ai_message(ai_response.content)
    
    print("AI: ", ai_response.content)
    
    