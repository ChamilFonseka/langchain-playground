from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# Initialize the ChatOpenAI model
model = ChatOpenAI()

# Initialize chat history as a list to store the conversation context
chat_history = []

# Add a system message to the chat history to define the behavior of the AI
chat_history.append(SystemMessage(content="You are a helpful AI assistant."))

# Start a loop to simulate an ongoing chat with the user
while True:
    # Get input from the user
    user_input = input("User: ")

    # If the user types 'exit', break the loop and stop the chat
    if user_input.lower() == "exit":
        break

    # Append the user's input as a HumanMessage to the chat history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke the model with the chat history to generate a response
    ai_reply = model.invoke(chat_history)

    # Print the AI's response
    print("AI:", ai_reply.content)

    # Append the AI's response as an AIMessage to the chat history
    chat_history.append(AIMessage(content=ai_reply.content))
