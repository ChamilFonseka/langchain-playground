from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema.runnable import RunnableLambda

load_dotenv()

# Initialize the model
model = ChatOpenAI()

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("humen", "Tell me {joke_count} jokes.")
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper())

# Define the chain 
chain = prompt_template | model | StrOutputParser() | uppercase_output

result = chain.invoke({"topic": "chicken", "joke_count": "3"})
print(result)
