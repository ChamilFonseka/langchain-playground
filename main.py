from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()

reply = llm.invoke("Hello how are you?")

print(reply)
