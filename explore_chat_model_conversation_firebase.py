import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

# Initialize the ChatOpenAI model
model = ChatOpenAI()

# Initialize Firestore
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'firebase-cred.json'
client = firestore.Client()

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id="user-session-id", 
    collection="langchain-chat-history",
    client=client
)

print("Chat history initialized.")
# Print the current chat history
print("Current chat history:", chat_history.messages)

while True:
    # Get input from the user
    user_input = input("You: ")

    # If the user types 'exit', break the loop and stop the chat
    if user_input.lower() == "exit":
        break

    # Append the user's input as a HumanMessage to the chat history
    chat_history.add_user_message(user_input)
    
    # Get the AI's reply
    ai_reply = model.invoke(chat_history.messages)

    # Append the AI's reply as an AIMessage to the chat history
    chat_history.add_ai_message(ai_reply.content)

    # Print the AI's reply
    print("AI:", ai_reply.content)


