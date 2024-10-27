from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

load_dotenv()

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate an AI image based on the following description: {image_desc}",
)
chain = prompt=prompt | llm

image_url = DallEAPIWrapper().run(chain.invoke("a cat with wings"))
print(image_url)