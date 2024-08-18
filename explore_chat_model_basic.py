from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Initialize the model
model = ChatOpenAI()

# User's first message
user_message_1 = "Hello, my name is Chamil"
print("User:", user_message_1)

# Generate AI reply for the first message
ai_reply_1 = model.invoke(user_message_1)
print("AI:", ai_reply_1.content)

# User's second message
user_message_2 = "What is my name?"
print("User:", user_message_2)

# Create the conversation history
conversation_history = [
    HumanMessage(content=user_message_1), 
    AIMessage(content=ai_reply_1.content),
    HumanMessage(content=user_message_2)
]

# Generate AI reply for the second message
ai_reply_2 = model.invoke(conversation_history)
print("AI:", ai_reply_2.content)