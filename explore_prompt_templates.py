from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the model
model = ChatOpenAI()

#Prompt with a single placeholder
print("---Prompt with a single placeholder---")
template = "Tell me joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "chicken"})
result = model.invoke(prompt)
print(result.content)

#Prompt with multiple placeholders
print("---Prompt with multiple placeholders---")
template = """You are a helpfull assistant.
Humen: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"adjective": "funny", "animal": "cat"})
result = model.invoke(prompt)
print(result.content)

#Prompt with System and Humen messages
print("---Prompt with System and Humen messages---")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("user", "Tell me {joke_count} jokes.")
]
prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topic": "chicken", "joke_count": "3"})
result = model.invoke(prompt)
print(result.content)



